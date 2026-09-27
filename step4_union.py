# step 4: candidates = name channel UNION addr channel, then cut each list
#
# cut rule: keep a neighbour if score >= RATIO * (best score of that s1 in that channel),
# and at most CAP per channel. ratio 0 = plain top CAP
#
# france isnt in train, so pick the setting on one country and test on the other
#
# python step4_union.py --sample 50000     uses cand_*_sample.parquet
# python step4_union.py                    uses the full cand_*.parquet

import argparse

import numpy as np
import pandas as pd

from common import CLEAN, load_clean, load_truth, pick_s1

RATIOS = [0.0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
CAPS = [5, 10, 20, 50]
KEEP = 0.97   # "good enough" = keeps 97% of the best recall we can get


def load_cands(field, sample):
    c = pd.read_parquet(CLEAN / (f"cand_{field}_sample.parquet" if sample else f"cand_{field}.parquet"))
    c["best"] = c.groupby("s1").score.transform("max")
    return c


def union(chans, ratio, cap, n):
    # one int per pair (s1 * n + other), duplicates across channels dropped
    keys = []
    for c in chans:
        keep = c[(c["rank"] < cap) & (c.score >= ratio * c.best)]
        keys.append(keep.s1.to_numpy() * n + keep.other.to_numpy())
    return np.unique(np.concatenate(keys))


def score(keys, true, true_country, s1, n):
    found = np.isin(true, keys)
    per_s1 = np.bincount(keys // n, minlength=n)[s1.index.to_numpy()]
    res = {}
    for c in sorted(s1.country.unique()):
        sizes = per_s1[(s1.country == c).to_numpy()]
        res[c] = dict(recall=found[true_country == c].mean(), avg=sizes.mean(), max=sizes.max())
    return res, found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    df = load_clean()
    n = len(df)
    s1 = pick_s1(df, args.sample)
    gt = load_truth(pd.Series(df.index, index=df.entity_id))
    gt = gt[gt.s1.isin(s1.index)]
    true = gt.s1.to_numpy() * n + gt.other.to_numpy()
    true_country = df.country.to_numpy()[gt.s1.to_numpy()]
    chans = [load_cands("name", args.sample), load_cands("addr", args.sample)]

    rows = []
    for ratio in RATIOS:
        for cap in CAPS:
            res, _ = score(union(chans, ratio, cap, n), true, true_country, s1, n)
            rows += [dict(ratio=ratio, cap=cap, country=c, **r) for c, r in res.items()]
    grid = pd.DataFrame(rows)
    print("== name + addr union: recall / avg candidates per s1 ==")
    print(grid.pivot_table(index=["ratio", "cap"], columns="country", values=["recall", "avg"]).round(3).to_string())

    # tune on one country, look at the other
    print(f"\n== cheapest setting that keeps {KEEP:.0%} of max recall ==")
    for c in grid.country.unique():
        g = grid[grid.country == c]
        best = g[g.recall >= KEEP * g.recall.max()].sort_values("avg").iloc[0]
        print(f"\ntuned on {c}: ratio={best.ratio}, cap={int(best.cap)}")
        pick = grid[(grid.ratio == best.ratio) & (grid.cap == best.cap)]
        print(pick[["country", "recall", "avg", "max"]].round(3).to_string(index=False))

    # what the last pick still misses
    _, found = score(union(chans, best.ratio, int(best.cap), n), true, true_country, s1, n)
    miss = gt[~found]
    addr = df.addr_norm.to_numpy()
    empty = (addr[miss.s1] == "") | (addr[miss.other] == "")
    print(f"\n{len(miss)} missed pairs, {empty.mean():.0%} have an empty address on one side\n")
    for i, j in miss.sample(min(10, len(miss)), random_state=0).itertuples(index=False):
        for r in (i, j):
            print(f"{df.entity_id[r]:>14} | {df.name_norm[r]} | {df.addr_norm[r]}")
        print()
