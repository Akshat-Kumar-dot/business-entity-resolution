# matcher step 1: numbers for every candidate pair the blocking kept
#
# the matcher will learn "same business or not" from these. groups:
#   name     - jaro-winkler, plain / token-set / partial ratio, rare-word overlap, ...
#   address  - same, plus house number agree/clash, digits overlap (NaN when an address is empty)
#   legal    - llc vs inc, pvt ltd vs llp... same form is a small plus, a clash a small minus
#   blocking - scores, ranks, gaps, keep probability from step 3/5
#   context  - how many candidates this s1 has in that source, is this one the best
#
# "rare-word overlap" weighs words by how rare they are, so a shared "patna" or "llc"
# counts for little and a shared "kharghar" or "orbivera" counts a lot
#
# pip install rapidfuzz
# python step6_features.py --sample 50000     -> clean/pairs_sample.parquet
# python step6_features.py                    -> clean/pairs.parquet  (same pairs as candidate_pairs.tsv)
# python step6_features.py --sample 50000 --keep 0.01   give the matcher a wider set (more recall)

import argparse
import math
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from expand import expand, make_keys
from common import CLEAN, load_truth, pick_s1
from step5_candidates import MODEL, full_chunks, keep, pair_table

T = 0.05   # keep threshold: 0.05 = exactly the pairs in candidate_pairs.tsv, lower = more for the matcher
LEGAL = (r"\b(?:llc|llp|pllc|inc|corp|corporation|incorporated|co|company|ltd|limited|pvt|private"
         r"|plc|pc|pa|lp|opc|sarl|sas|sasu|sa|eurl|sci|snc)\b")


def load_records():
    df = pd.read_parquet(CLEAN / "records.parquet",
                         columns=["entity_id", "source", "country", "business_name",
                                  "name_norm", "addr_norm", "house_no"])
    # legal form from the raw name (normalize.py strips it from name_norm)
    raw = df.business_name.str.lower().str.replace(r"(?<=\b[a-z])\.", "", regex=True)
    df["legal"] = raw.str.findall(LEGAL).map(frozenset)
    return df


def rarity(texts):
    # idf of every word: log(n / records containing it)
    c = Counter()
    for t in texts:
        c.update(set(t.split()))
    n = len(texts)
    return {w: math.log(n / k) for w, k in c.items()}


def rare_overlap(xs, ys, idf):
    # shared words weighted by rarity / all words weighted by rarity
    out = np.full(len(xs), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        a, b = set(x.split()), set(y.split())
        if a and b:
            union = sum(idf.get(w, 10) for w in a | b)
            out[i] = sum(idf.get(w, 10) for w in a & b) / union
    return out


def digits_overlap(xs, ys):
    out = np.full(len(xs), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        a = {w for w in x.split() if any(ch.isdigit() for ch in w)}
        b = {w for w in y.split() if any(ch.isdigit() for ch in w)}
        if a and b:
            out[i] = len(a & b) / len(a | b)
    return out


def sim(xs, ys, scorer):
    return process.cpdist(xs, ys, scorer=scorer, workers=-1).astype(np.float32)


def features(p, rec, idf_name, idf_addr):
    a, b = p.s1.to_numpy(), p.other.to_numpy()
    na, nb = rec.name_norm.to_numpy()[a], rec.name_norm.to_numpy()[b]
    aa, ab = rec.addr_norm.to_numpy()[a], rec.addr_norm.to_numpy()[b]
    f = p.reset_index(drop=True)

    # name
    f["n_jw"] = sim(na, nb, JaroWinkler.normalized_similarity)
    f["n_ratio"] = sim(na, nb, fuzz.ratio) / 100
    f["n_tset"] = sim(na, nb, fuzz.token_set_ratio) / 100
    f["n_partial"] = sim(na, nb, fuzz.partial_ratio) / 100
    f["n_rare"] = rare_overlap(na, nb, idf_name)
    la, lb = np.char.str_len(na.astype(str)), np.char.str_len(nb.astype(str))
    f["n_len_diff"] = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1)
    f["n_first_eq"] = [x.split()[:1] == y.split()[:1] for x, y in zip(na, nb)]

    # address (NaN when one side is empty: "unknown", not "different")
    empty = (aa == "") | (ab == "")
    f["a_empty"] = empty
    for col, scorer in (("a_ratio", fuzz.ratio), ("a_tset", fuzz.token_set_ratio)):
        f[col] = np.where(empty, np.nan, sim(aa, ab, scorer) / 100)
    f["a_rare"] = rare_overlap(aa, ab, idf_addr)
    f["a_digits"] = digits_overlap(aa, ab)
    ha, hb = rec.house_no.to_numpy()[a], rec.house_no.to_numpy()[b]
    f["house"] = np.select([(ha == "") | (hb == ""), ha == hb], [0, 1], -1)   # 0 unknown, 1 same, -1 clash

    # legal form
    ga, gb = rec.legal.to_numpy()[a], rec.legal.to_numpy()[b]
    f["legal_same"] = [bool(x & y) for x, y in zip(ga, gb)]
    f["legal_clash"] = [bool(x) and bool(y) and not (x & y) for x, y in zip(ga, gb)]

    # context: other candidates of this s1 in the same source
    f["is_s3"] = rec.source.to_numpy()[b] == "S3"
    grp = f.groupby(["s1", "is_s3"]).keep_p
    f["n_same_src"] = grp.transform("size")
    f["best_in_src"] = grp.rank(ascending=False, method="first") == 1
    f["gap_to_best"] = grp.transform("max") - f.keep_p

    f["country"] = rec.country.to_numpy()[a]
    return f


def add_label(f, true_keys, n):
    f["y"] = np.isin(f.s1.to_numpy() * n + f.other.to_numpy(), true_keys).astype(np.int8)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--keep", type=float, default=T)
    ap.add_argument("--no-expand", action="store_true", help="skip cluster expansion")
    args = ap.parse_args()
    T = args.keep

    rec = load_records()
    n = len(rec)
    print("word rarity..", flush=True)
    idf_name, idf_addr = rarity(rec.name_norm.to_numpy()), rarity(rec.addr_norm.to_numpy())
    gt = load_truth(pd.Series(rec.index, index=rec.entity_id))
    true_keys = gt.s1.to_numpy() * n + gt.other.to_numpy()
    m = joblib.load(MODEL)
    keys = None if args.no_expand else make_keys(rec)

    def kept(p):   # blocking keep rule + cluster expansion = exactly the pairs in candidate_pairs.tsv
        p = keep(p, m, T)
        return p if keys is None else expand(p, keys)

    if args.sample:
        p = pair_table(rec.source.to_numpy(),
                       pd.read_parquet(CLEAN / "cand_name_sample.parquet"),
                       pd.read_parquet(CLEAN / "cand_addr_sample.parquet"))
        out = add_label(features(kept(p), rec, idf_name, idf_addr), true_keys, n)
        path = CLEAN / "pairs_sample.parquet"
        s1 = pick_s1(rec, args.sample)
        print(f"blocking recall on this sample: "
              f"{out.y.sum() / gt.s1.isin(s1.index).sum():.3f}")
    else:
        parts = [add_label(features(kept(p), rec, idf_name, idf_addr), true_keys, n)
                 for _, p in full_chunks(rec)]
        out = pd.concat(parts, ignore_index=True)
        path = CLEAN / "pairs.parquet"

    out.to_parquet(path, index=False)
    print(f"saved {path}: {len(out):,} pairs, {out.y.mean():.1%} true, {out.shape[1]} columns")
    print("\n== average of each feature, false pairs (0) vs true pairs (1) ==")
    print(out.drop(columns=["s1", "other", "country"]).groupby("y").mean().T.round(3))
