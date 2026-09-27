# step 5: decide which candidates to keep + write candidate_pairs.tsv
#
# every (s1, other) pair found by the name or addr channel gets a few numbers:
#   - name score, addr score (0 if that channel didnt find it)
#   - rank inside its own source. s2 and s3 are ranked apart, so a pile of
#     s2 lookalikes cant push the real s3 match out
#   - gap = how far below the best score of that s1 (same channel, same source)
#   - both = found by both channels (strong hint)
# a small tree model (gradient boosting) turns that into p(match), we keep pairs with p >= T.
# trees, not a straight line: rank 0 vs 1 matters way more than 30 vs 31, and
# 'channel didnt find it' is its own case
#
# the model only has 7 numbers to learn, so the 50k sample is plenty to train it.
# france check: train on one country, test on the other.
#
# python step5_candidates.py --sample 50000         train + pick T on the sample, saves the model
# python step5_candidates.py --threshold 0.01       apply to the full files -> candidate_pairs.tsv
#                                                   (+ cluster expansion, see expand.py; --no-expand to skip)

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from expand import expand, make_keys
from common import CLEAN, SPLIT, SUBMIT, TRAINED, load_clean, load_truth, pick_s1

FEATURES = ["name_score", "addr_score", "name_rank", "addr_rank", "name_gap", "addr_gap", "both"]
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
NO_RANK = 50          # rank we give when a channel didnt find the pair
CHUNK = 200_000       # s1 records per pass on the full files
MODEL = TRAINED / "keep_model.pkl"   # trained on train, reused on test


def new_model():
    return HistGradientBoostingClassifier(max_iter=200, random_state=0)


def pair_table(src_of, name, addr):
    # one row per (s1, other) found by either channel, with the features above
    chans = []
    for f, c in (("name", name), ("addr", addr)):
        c = c[["s1", "other", "score"]].copy()
        c["src"] = src_of[c.other.to_numpy()]
        c = c.sort_values(["s1", "src", "score"], ascending=[True, True, False])
        c[f"{f}_rank"] = c.groupby(["s1", "src"]).cumcount()
        c[f"{f}_gap"] = c.groupby(["s1", "src"]).score.transform("max") - c.score
        chans.append(c.drop(columns="src").rename(columns={"score": f"{f}_score"}))

    p = chans[0].merge(chans[1], on=["s1", "other"], how="outer")
    p["both"] = (p.name_score.notna() & p.addr_score.notna()).astype(int)
    return p.fillna({"name_score": 0, "addr_score": 0, "name_rank": NO_RANK, "addr_rank": NO_RANK,
                     "name_gap": 1, "addr_gap": 1})


def show(p, keep, true_keys, s1, n):
    # recall + candidates per s1, per country
    kept = p[keep]
    found = np.isin(true_keys["key"], kept.s1.to_numpy() * n + kept.other.to_numpy())
    per_s1 = kept.groupby("s1").size().reindex(s1.index, fill_value=0)
    out = {}
    for c in sorted(s1.country.unique()):
        sizes = per_s1[(s1.country == c).to_numpy()]
        out[c] = f"recall {found[true_keys['country'] == c].mean():.3f}  avg {sizes.mean():5.2f}  max {sizes.max()}"
    return out


def train_and_pick(df, sample):
    n = len(df)
    s1 = pick_s1(df, sample)
    gt = load_truth(pd.Series(df.index, index=df.entity_id))
    gt = gt[gt.s1.isin(s1.index)]
    true = {"key": gt.s1.to_numpy() * n + gt.other.to_numpy(),
            "country": df.country.to_numpy()[gt.s1.to_numpy()]}

    p = pair_table(df.source.to_numpy(),
                   pd.read_parquet(CLEAN / "cand_name_sample.parquet"),
                   pd.read_parquet(CLEAN / "cand_addr_sample.parquet"))
    p["y"] = np.isin(p.s1.to_numpy() * n + p.other.to_numpy(), true["key"]).astype(int)
    p["country"] = df.country.to_numpy()[p.s1.to_numpy()]
    print(f"{len(p):,} candidate pairs, {p.y.mean():.1%} of them true\n")

    print("== baseline: keep top k per source in either channel ==")
    for k in (1, 2, 3, 5, 10):
        print(f"k={k:<3}", show(p, (p.name_rank < k) | (p.addr_rank < k), true, s1, n))

    # train on one country, test on the other
    countries = sorted(p.country.unique())
    for tr in countries:
        te = [c for c in countries if c != tr]
        m = new_model()
        m.fit(p.loc[p.country == tr, FEATURES], p.loc[p.country == tr, "y"])
        prob = m.predict_proba(p[FEATURES])[:, 1]
        print(f"\n== learned keep rule, trained on {tr} only (look at {', '.join(te)}) ==")
        for t in THRESHOLDS:
            print(f"T={t:<5}", show(p, prob >= t, true, s1, n))

    # final model: both countries
    joblib.dump(new_model().fit(p[FEATURES], p.y), MODEL)
    print(f"\nsaved {MODEL}")


def full_chunks(df):
    # the full candidate files, one chunk of s1 at a time so memory stays small
    s1 = df[df.source == "S1"]
    for lo in range(0, len(s1), CHUNK):
        rows = s1.index[lo:lo + CHUNK]
        filt = [("s1", ">=", rows.min()), ("s1", "<=", rows.max())]
        p = pair_table(df.source.to_numpy(),
                       pd.read_parquet(CLEAN / "cand_name.parquet", filters=filt),
                       pd.read_parquet(CLEAN / "cand_addr.parquet", filters=filt))
        yield rows, p
        print(f"  {min(lo + CHUNK, len(s1)):,} / {len(s1):,} s1 done", flush=True)


def keep(p, m, threshold):
    # pairs the blocking keeps. also used by the matcher, so it sees the exact same pairs
    p = p.assign(keep_p=m.predict_proba(p[FEATURES])[:, 1])
    return p[p.keep_p >= threshold]


def write_candidates(df, threshold, use_expand=True):
    # full files, a chunk of s1 at a time so memory stays small
    m = joblib.load(MODEL)
    keys = make_keys(df) if use_expand else None
    ids = df.entity_id.to_numpy()
    SUBMIT.mkdir(exist_ok=True)
    out = SUBMIT / "candidate_pairs.tsv" if SPLIT == "test" else CLEAN.parent / "candidate_pairs.tsv"
    total, n_s1 = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for rows, p in full_chunks(df):
            p = keep(p, m, threshold)
            if use_expand:
                p = expand(p, keys)
            total, n_s1 = total + len(p), n_s1 + len(rows)
            lists = p.groupby("s1").other.apply(lambda o: ",".join(ids[o.to_numpy()]))
            for r in rows:   # every s1 gets a line, even with no candidates
                f.write(f"{ids[r]}\t{lists.get(r, '')}\n")
    print(f"saved {out}: {total:,} pairs, {total / n_s1:.2f} per s1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-expand", action="store_true", help="skip cluster expansion")
    args = ap.parse_args()

    df = load_clean()
    if args.sample:
        train_and_pick(df, args.sample)
    else:
        write_candidates(df, args.threshold, not args.no_expand)
