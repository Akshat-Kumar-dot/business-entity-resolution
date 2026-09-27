# cluster expansion: the true matches of one s1 are near-copies of each other in s2/s3.
# blocking often finds some of them but misses others (empty address, garbled name).
# so: take the candidates blocking is already sure about ("anchors") and pull in every
# s2/s3 record that shares a typo-tolerant key with an anchor.
#
# only hash joins on keys -> linear, scales to billions. keys shared by more than
# KEY_CAP records are dropped (a key like "subway" would drag in half the country).
# the new pairs go through the matcher like everything else (marked expanded=1).

import numpy as np
import pandas as pd

KEY_CAP = 20        # ignore keys used by more records than this
ANCHOR_P = 0.5      # keep probability a candidate needs to count as an anchor


def sorted_words(s):
    return s.str.split().map(lambda w: " ".join(sorted(set(w))))


def prefixes(s, n=4):          # "consultants limitet" -> "cons limi"   (survives typos after letter 4)
    return s.str.split().map(lambda w: " ".join(sorted({x[:n] for x in w})))


def digits(s):                 # just the numbers of an address
    return s.str.split().map(lambda w: " ".join(sorted({x for x in w if any(ch.isdigit() for ch in x)})))


def make_keys(rec):
    # one row per (s2/s3 record, key). keys are hashed to int64 to save memory,
    # and carry the country so expansion never crosses countries
    oth = rec[rec.source != "S1"]
    num = digits(oth.addr_norm)
    kinds = {
        "name_words": sorted_words(oth.name_norm),
        "addr_words": sorted_words(oth.addr_norm),
        "name_prefix4": prefixes(oth.name_norm),
        "digits_name3": np.where(num != "", num + "|" + oth.name_norm.str[:3], ""),
    }
    parts = []
    for kind, k in kinds.items():
        k = pd.Series(k, index=oth.index)
        k = k[k != ""]
        h = pd.util.hash_array((kind + "|" + oth.country[k.index] + "|" + k).to_numpy().astype(str))
        t = pd.DataFrame({"row": k.index.to_numpy(), "key": h.astype(np.int64)})
        parts.append(t[t.groupby("key").row.transform("size") <= KEY_CAP])
    return pd.concat(parts, ignore_index=True)


def expand(p, keys):
    # p: kept pairs (with keep_p). returns p + new pairs reached through an anchor
    anchors = p.loc[p.keep_p >= ANCHOR_P, ["s1", "other", "keep_p"]]
    reach = anchors.merge(keys, left_on="other", right_on="row")[["s1", "key", "keep_p"]]
    new = reach.merge(keys, on="key")[["s1", "row", "keep_p"]].rename(columns={"row": "other"})
    new = (new.groupby(["s1", "other"])
              .agg(anchor_p=("keep_p", "max"), n_links=("keep_p", "size"))
              .reset_index())
    already = pd.MultiIndex.from_frame(p[["s1", "other"]])
    new = new[~pd.MultiIndex.from_frame(new[["s1", "other"]]).isin(already)]

    old = p.assign(expanded=0, anchor_p=np.nan, n_links=0)
    return pd.concat([old, new.assign(expanded=1)], ignore_index=True)
