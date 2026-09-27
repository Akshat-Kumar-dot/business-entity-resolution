# matcher: two lightgbm stages -> same business or not
#
# stage 1 judges every (s1, candidate) pair on its own (the step6 features)
# stage 2 also looks at the other candidates of the same s1:
#   - best / second best stage-1 probability in the list, how many are strong
#   - this pair's rank and gap to the best
#   - how similar this candidate is to the s1's most likely match (the "anchor").
#     true matches of one s1 are duplicates of each other, so a look-alike of the
#     anchor is probably true too; and if even the best is weak -> probably a singleton
#
# the official score: f0.5 per source-1 record, averaged over ALL s1 (singletons too).
#   s1 with no true match: empty prediction = 1.0, any prediction = 0.0
#   pairs blocking lost count as misses -> this is what the leaderboard sees
#
# honest numbers, two ways:
#   cv      - 5 folds split by s1: every s1 is scored by models that never saw it
#   cross   - train on one country only, score the other (our stand-in for unseen france)
#
# python step7_match.py --sample 50000     validate, pick threshold, save both models
# python step7_match.py --apply 0.6        full pairs (or test with SPLIT=test) -> matching_results.tsv

import argparse

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.model_selection import GroupKFold

from common import CLEAN, ROOT, SPLIT, SUBMIT, TRAINED, load_clean, load_truth, pick_s1

MODEL = TRAINED / "matcher.pkl"
NOT_FEATURES = ["s1", "other", "country", "y"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# stage 2 was +0.3..0.5 in cv but -2..-3.5 on an unseen country (it leans on stage-1
# probabilities, which shift on new data). the test set has unseen france, so: off.
USE_STAGE2 = False


def new_model():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=0, verbose=-1)


def list_features(pairs, p1, rec):
    # stage 2 extras: how this candidate sits among the other candidates of its s1
    d = pd.DataFrame({"s1": pairs.s1.to_numpy(), "other": pairs.other.to_numpy(), "p1": p1})
    d = d.sort_values(["s1", "p1"], ascending=[True, False])
    g = d.groupby("s1")
    d["p1_rank"] = g.cumcount()
    second = d.p1_rank == 1
    d["p1_best"] = g.p1.transform("max")
    d["p1_second"] = d.p1.where(second).groupby(d.s1).transform("max").fillna(0.0)
    d["p1_gap"] = d.p1_best - d.p1
    d["n_strong"] = (d.p1 >= 0.5).groupby(d.s1).transform("sum")

    # anchor = best candidate of this s1 (for the best one itself: the second best)
    top = g.other.transform("first").to_numpy()
    runner_up = d.other.where(second).groupby(d.s1).transform("max").fillna(-1).astype(int).to_numpy()
    o = d.other.to_numpy()
    is_top = o == top
    anchor = np.where(is_top, runner_up, top)
    has = anchor >= 0
    a = np.where(has, anchor, o)          # placeholder when there is no anchor, masked below
    name, addr, src = rec.name_norm.to_numpy(), rec.addr_norm.to_numpy(), rec.source.to_numpy()

    d["anchor_p1"] = np.where(has, np.where(is_top, d.p1_second, d.p1_best), np.nan)
    d["anchor_name"] = np.where(has, process.cpdist(name[o], name[a], scorer=JaroWinkler.normalized_similarity, workers=-1), np.nan)
    d["anchor_addr"] = np.where(has, process.cpdist(addr[o], addr[a], scorer=fuzz.token_set_ratio, workers=-1) / 100, np.nan)
    d["anchor_same_src"] = np.where(has, src[o] == src[a], np.nan)

    return d.sort_index().drop(columns=["s1", "other"])   # back in the original pair order


def per_s1_f05(pred, gt, s1_rows):
    # official metric: one f0.5 per s1 (average it for the score)
    n_pred = pred.groupby("s1").size().reindex(s1_rows, fill_value=0).to_numpy()
    n_true = gt.groupby("s1").size().reindex(s1_rows, fill_value=0).to_numpy()
    hit = pd.MultiIndex.from_frame(pred[["s1", "other"]]).isin(pd.MultiIndex.from_frame(gt[["s1", "other"]]))
    n_hit = pd.Series(hit, index=pred.s1.to_numpy()).groupby(level=0).sum().reindex(s1_rows, fill_value=0).to_numpy()

    p = n_hit / np.maximum(n_pred, 1)
    r = n_hit / np.maximum(n_true, 1)
    f = np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    f[(n_pred == 0) & (n_true == 0)] = 1.0      # singleton, correctly left empty
    return f


def table(pairs, prob, gt, s1, how):
    rows = []
    for c in sorted(s1.country.unique()):
        here = (pairs.country == c).to_numpy()
        s1_c = s1.index[s1.country == c]
        gt_c = gt[gt.s1.isin(s1_c)]
        for t in THRESHOLDS:
            pred = pairs.loc[here & (prob >= t), ["s1", "other"]]
            rows.append(dict(how=how, country=c, T=t, f05=per_s1_f05(pred, gt_c, s1_c).mean()))
    return pd.DataFrame(rows)


def breakdown(pairs, prob, t, gt, s1):
    # where do we lose points? per-s1 score by number of true matches, and by blocking coverage
    pred = pairs.loc[prob >= t, ["s1", "other"]]
    f = pd.Series(per_s1_f05(pred, gt, s1.index), index=s1.index)
    n_true = gt.groupby("s1").size().reindex(s1.index, fill_value=0)
    found = pd.MultiIndex.from_frame(gt[["s1", "other"]]).isin(pd.MultiIndex.from_frame(pairs[["s1", "other"]]))
    lost = (~pd.Series(found, index=gt.s1.to_numpy())).groupby(level=0).any().reindex(s1.index, fill_value=False)
    d = pd.DataFrame({"country": s1.country, "true_matches": n_true.clip(upper=4).astype(str).str.replace("4", "4+"),
                      "blocking_lost_some": lost, "f05": f})
    print(d.groupby(["country", "true_matches"]).f05.agg(["mean", "size"]).round(3).to_string())
    print()
    print(d.groupby(["country", "blocking_lost_some"]).f05.agg(["mean", "size"]).round(3).to_string())


def out_of_fold(X, y, groups):
    # every row predicted by a model that never saw its s1
    p = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
        p[te] = new_model().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


def validate(sample):
    rec = load_clean()
    s1 = pick_s1(rec, sample)
    gt = load_truth(pd.Series(rec.index, index=rec.entity_id))
    gt = gt[gt.s1.isin(s1.index)]

    pairs = pd.read_parquet(CLEAN / "pairs_sample.parquet")
    X1, y = pairs.drop(columns=NOT_FEATURES), pairs.y
    by_country = {c: (pairs.country == c).to_numpy() for c in pairs.country.unique()}

    print("stage 1..", flush=True)
    cv1 = out_of_fold(X1, y, pairs.s1)
    cross1 = np.zeros(len(pairs))
    for c, here in by_country.items():
        cross1[here] = new_model().fit(X1[~here], y[~here]).predict_proba(X1[here])[:, 1]

    print("stage 2..", flush=True)
    X2 = pd.concat([X1, list_features(pairs, cv1, rec)], axis=1)
    cv2 = out_of_fold(X2, y, pairs.s1)
    cross2 = np.zeros(len(pairs))
    X2_cross = pd.concat([X1, list_features(pairs, cross1, rec)], axis=1)   # what a new country would look like
    for c, here in by_country.items():
        cross2[here] = new_model().fit(X2[~here], y[~here]).predict_proba(X2_cross[here])[:, 1]

    res = pd.concat([table(pairs, cv1, gt, s1, "cv stage1"), table(pairs, cv2, gt, s1, "cv stage2"),
                     table(pairs, cross1, gt, s1, "cross stage1"), table(pairs, cross2, gt, s1, "cross stage2")])
    print("\n== official score: per-s1 f0.5 (singletons in) ==")
    print(res.pivot_table(index="T", columns=["how", "country"], values="f05").round(4).to_string())

    final = "cv stage2" if USE_STAGE2 else "cv stage1"
    best_t = res[res.how == final].groupby("T").f05.mean().idxmax()
    print(f"\nbest threshold on {final} (avg of countries): T={best_t}")
    print(res[res["T"] == best_t].pivot_table(index="how", columns="country", values="f05").round(4).to_string())

    print(f"\n== where points are lost ({final}, T={best_t}) ==")
    breakdown(pairs, cv2 if USE_STAGE2 else cv1, best_t, gt, s1)

    # final models on everything. stage 2 learns from the out-of-fold stage-1 numbers,
    # the same kind of numbers it will get on new data
    m1 = new_model().fit(X1, y)
    m2 = new_model().fit(X2, y)
    joblib.dump({"stage1": m1, "stage2": m2}, MODEL)
    used = m2 if USE_STAGE2 else m1
    imp = pd.Series(used.booster_.feature_importance("gain"), index=used.feature_name_)
    print(f"\nsaved {MODEL}\n\n== what the final model leans on (share of gain) ==")
    print((imp / imp.sum()).sort_values(ascending=False).round(3).head(12).to_string())


def apply(threshold):
    rec = load_clean()
    ids = rec.entity_id.to_numpy()
    pairs = pd.read_parquet(CLEAN / "pairs.parquet")
    m = joblib.load(MODEL)
    X1 = pairs.drop(columns=NOT_FEATURES)
    p = m["stage1"].predict_proba(X1)[:, 1]
    if USE_STAGE2:
        p = m["stage2"].predict_proba(pd.concat([X1, list_features(pairs, p, rec)], axis=1))[:, 1]
    pred = pairs.loc[p >= threshold, ["s1", "other"]]

    s1 = rec[rec.source == "S1"]
    gt = load_truth(pd.Series(rec.index, index=rec.entity_id))
    if len(gt):   # train: score it (the models saw only the 50k sample, the rest is new to them)
        f = per_s1_f05(pred, gt, s1.index)
        print(pd.Series(f, index=s1.index).groupby(s1.country).mean().round(4).to_string())

    SUBMIT.mkdir(exist_ok=True)
    out = SUBMIT / "matching_results.tsv" if SPLIT == "test" else ROOT / "matching_results_train.tsv"
    lists = pred.groupby("s1").other.apply(lambda o: ",".join(ids[o.to_numpy()]))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for r in s1.index:   # every s1 gets a line, even with no match
            fh.write(f"{ids[r]}\t{lists.get(r, '')}\n")
    print(f"saved {out}: {len(pred):,} matches, {len(pred) / len(s1):.2f} per s1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--apply", type=float, default=None, help="threshold")
    args = ap.parse_args()

    if args.sample:
        validate(args.sample)
    else:
        apply(args.apply)
