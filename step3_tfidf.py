# step 3: fuzzy blocking. for each s1 -> top k most similar s2/s3 in the same country
#
# two channels:
#   name = name 3-grams + 0.5 * address words    (~50 shops share a name, the address breaks the tie)
#   addr = address 3-grams + 0.5 * name 3-grams  (offices share an address, the name breaks the tie)
#
# has to scale to billions, so:
#   - hashing vectorizer -> no vocab kept in memory
#   - s2/s3 done in shards, top k per shard then merge. exact: the best k overall
#     are always inside some shard's top k
#   - a query skips pieces found in > MAX_HITS records (they match everything anyway)
#     but always keeps its 3 rarest. work per query is capped -> total work is linear
#
# pip install scikit-learn sparse_dot_topn
# python step3_tfidf.py --sample 50000         quick check on 50k s1
# python step3_tfidf.py                        full run, name channel
# python step3_tfidf.py --field addr           full run, addr channel
# python step3_tfidf.py --sample 50000 --max-hits 50000     try another cap

import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from common import CLEAN, load_clean, load_truth, pick_s1

K = 50                 # keep top 50, smaller k are read off the rank later
KS = [5, 10, 20, 50]
MIN_SCORE = 0.1
Q_CHUNK = 100_000      # s1 rows per multiply
SHARD = 5_000_000      # s2/s3 rows in memory at once
MAX_HITS = 50_000      # fixed number, not a % -> work per query stays capped at any data size
KEEP_RAREST = 3
WIDTH = 2**20          # hash buckets per part

HASH = {
    "char": HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=WIDTH,
                              alternate_sign=False, norm=None, dtype=np.float32),
    "word": HashingVectorizer(analyzer="word", n_features=WIDTH,
                              alternate_sign=False, norm=None, dtype=np.float32),
}
CHANNELS = {   # (column, piece type, weight)
    "name": [("name_norm", "char", 1.0), ("addr_norm", "word", 0.5)],
    "addr": [("addr_norm", "char", 1.0), ("name_norm", "char", 0.5)],
}


def shards(df):
    for i in range(0, len(df), SHARD):
        yield i, df.iloc[i:i + SHARD]


def count_hits(oth, parts):
    # how many s2/s3 records each piece shows up in, one pass over the shards
    hits = [np.zeros(WIDTH, dtype=np.int64) for _ in parts]
    for _, sh in shards(oth):
        for h, (col, kind, _) in zip(hits, parts):
            h += np.bincount(HASH[kind].transform(sh[col]).indices, minlength=WIDTH)
    return hits


def vectors(df, parts, idfs):
    # each part weighted + scaled to length 1 on its own first, so a long address
    # cant drown out a short name. then mix by weight
    blocks = [w * normalize(HASH[kind].transform(df[col]) @ sp.diags(idf))
              for idf, (col, kind, w) in zip(idfs, parts)]
    return normalize(sp.hstack(blocks).tocsr())


def drop_common(q, hits):
    # zero out pieces seen > MAX_HITS times, but each row keeps its KEEP_RAREST rarest ones
    q = q.tocsr(copy=True)
    h = hits[q.indices]
    row = np.repeat(np.arange(q.shape[0]), np.diff(q.indptr))
    order = np.lexsort((h, row))                     # by row, rarest first
    pos = np.empty_like(order)
    pos[order] = np.arange(len(order)) - q.indptr[row[order]]
    q.data[(h > MAX_HITS) & (pos >= KEEP_RAREST)] = 0
    q.eliminate_zeros()
    return q


def block(s1, oth, field):
    parts = CHANNELS[field]
    if field == "addr":   # empty address cant match anything
        s1, oth = s1[s1.addr_norm != ""], oth[oth.addr_norm != ""]

    hits = count_hits(oth, parts)
    idfs = [(np.log((1 + len(oth)) / (1 + h)) + 1).astype(np.float32) for h in hits]
    q = drop_common(vectors(s1, parts, idfs), np.concatenate(hits))

    best = None
    for off, sh in shards(oth):
        x = vectors(sh, parts, idfs).T.tocsr()
        found = []
        for i in range(0, q.shape[0], Q_CHUNK):
            top = sp_matmul_topn(q[i:i + Q_CHUNK], x, top_n=K, threshold=MIN_SCORE, n_threads=-1).tocoo()
            found.append(pd.DataFrame({"s1": s1.index.to_numpy()[top.row + i],
                                       "other": oth.index.to_numpy()[top.col + off],
                                       "score": top.data}))
            print(f"    shard {off // SHARD + 1}: {min(i + Q_CHUNK, q.shape[0]):>9,} / {q.shape[0]:,}", flush=True)
        # merge right away -> never more than 2 shards of results in memory
        best = keep_top(found if best is None else [best, *found])
    return best


def keep_top(frames):
    cand = pd.concat(frames, ignore_index=True).sort_values(["s1", "score"], ascending=[True, False])
    cand["rank"] = cand.groupby("s1").cumcount()
    return cand[cand["rank"] < K].reset_index(drop=True)


def report(df, cand, gt, s1):
    n = len(df)
    true = gt.s1.to_numpy() * n + gt.other.to_numpy()
    true_country = df.country.to_numpy()[gt.s1.to_numpy()]
    rows = []
    for k in KS:
        top = cand[cand["rank"] < k]
        found = np.isin(true, top.s1.to_numpy() * n + top.other.to_numpy())
        per_s1 = top.groupby("s1").size().reindex(s1.index, fill_value=0)
        for c in sorted(s1.country.unique()):
            sizes = per_s1[(s1.country == c).to_numpy()]
            rows.append({"k": k, "country": c, "recall": round(found[true_country == c].mean(), 4),
                         "avg_cand": round(sizes.mean(), 2), "max_cand": sizes.max()})
    print(pd.DataFrame(rows).to_string(index=False))
    return found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="name", choices=list(CHANNELS))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max-hits", type=int, default=MAX_HITS)
    args = ap.parse_args()
    MAX_HITS = args.max_hits

    df = load_clean()
    gt = load_truth(pd.Series(df.index, index=df.entity_id))
    s1 = pick_s1(df, args.sample)
    gt = gt[gt.s1.isin(s1.index)]
    oth = df[df.source != "S1"]

    out = []
    for c in sorted(s1.country.unique()):
        t = time.time()
        print(f"\n{c}: {args.field}", flush=True)
        out.append(block(s1[s1.country == c], oth[oth.country == c], args.field))
        print(f"  done in {time.time() - t:.0f}s")
    cand = pd.concat(out, ignore_index=True)

    if len(gt):   # test set has no answers, nothing to report
        print(f"\n== {args.field}: recall / candidates per s1 ==")
        found = report(df, cand, gt, s1)

        print(f"\n== some pairs missed at k={K} ==")
        miss = gt[~found]
        for i, j in miss.sample(min(10, len(miss)), random_state=0).itertuples(index=False):
            for r in (i, j):
                print(f"{df.entity_id[r]:>14} | {df.name_norm[r]} | {df.addr_norm[r]}")
            print()

    path = CLEAN / (f"cand_{args.field}_sample.parquet" if args.sample else f"cand_{args.field}.parquet")
    cand.to_parquet(path, index=False)
    print("saved", path)
