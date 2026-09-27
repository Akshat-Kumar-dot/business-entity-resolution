# diagnostic: can "cluster expansion" win back the matches blocking loses?
#
# for every true match blocking MISSED, look at the true matches of the same s1 that
# blocking DID find ("siblings"). if the missed record is a near-copy of a found sibling,
# we can reach it through that sibling (s1 -> sibling -> missed record).
#
# python diagnose_missed.py      (uses clean/pairs_sample.parquet from step6)

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from common import CLEAN, load_clean, load_truth

rec = load_clean()
name, addr, ctry = rec.name_norm.to_numpy(), rec.addr_norm.to_numpy(), rec.country.to_numpy()
pairs = pd.read_parquet(CLEAN / "pairs_sample.parquet", columns=["s1", "other"])
gt = load_truth(pd.Series(rec.index, index=rec.entity_id))
gt = gt[gt.s1.isin(pairs.s1.unique())]

cand = set(zip(pairs.s1, pairs.other))
gt["found"] = [(a, b) in cand for a, b in zip(gt.s1, gt.other)]
print(f"true pairs: {len(gt):,}   found by blocking: {gt.found.mean():.3f}\n")

rows = []
found_by_s1 = gt[gt.found].groupby("s1").other.apply(list).to_dict()
for s1, miss in gt.loc[~gt.found, ["s1", "other"]].itertuples(index=False):
    sibs = found_by_s1.get(s1, [])
    best_n = max((JaroWinkler.normalized_similarity(name[miss], name[s]) for s in sibs), default=np.nan)
    best_a = max((fuzz.token_set_ratio(addr[miss], addr[s]) / 100 for s in sibs if addr[miss] and addr[s]), default=np.nan)
    rows.append(dict(country=ctry[s1], has_sibling=bool(sibs), sib_name=best_n, sib_addr=best_a,
                     s1_name=JaroWinkler.normalized_similarity(name[s1], name[miss]),
                     empty_addr=(addr[miss] == "") or (addr[s1] == "")))
m = pd.DataFrame(rows)

m["reachable"] = (m.sib_name >= 0.9) | (m.sib_addr >= 0.9)
print("== missed true matches ==")
print(m.groupby("country").agg(missed=("reachable", "size"),
                               has_found_sibling=("has_sibling", "mean"),
                               near_copy_of_sibling=("reachable", "mean"),
                               empty_address=("empty_addr", "mean")).round(3).to_string())

# what recall would be if expansion picked up every reachable one
gt_c = ctry[gt.s1.to_numpy()]
for c in sorted(m.country.unique()):
    now = gt.found[gt_c == c].mean()
    extra = m.reachable[m.country == c].sum() / (gt_c == c).sum()
    print(f"{c:6} blocking recall now {now:.3f}  -> with perfect expansion up to {now + extra:.3f}")

print("\n== how similar missed records are to their best found sibling ==")
print(m[["sib_name", "sib_addr", "s1_name"]].describe(percentiles=[.25, .5, .75]).round(3).to_string())


# --- can a cheap exact key reach them? (hash join = scales to billions, no similarity search)
# key = the words of the name (or address) in sorted order, so word order doesnt matter
def key(s):
    return " ".join(sorted(set(s.split())))

src = rec.source.to_numpy()
others = rec.index[src != "S1"]
nkey = pd.Series([key(name[i]) for i in others], index=others)
akey = pd.Series([key(addr[i]) for i in others], index=others)
nsize = nkey.map(nkey.value_counts())       # how many s2/s3 records share this key
asize = akey.map(akey.value_counts())

rows = []
for s1, miss in gt.loc[~gt.found, ["s1", "other"]].itertuples(index=False):
    sibs = found_by_s1.get(s1, [])
    by_name = any(nkey[miss] == nkey[s] for s in sibs)
    by_addr = addr[miss] != "" and any(akey[miss] == akey[s] for s in sibs)
    rows.append(dict(country=ctry[s1], by_name=by_name, by_addr=by_addr,
                     name_group=nsize[miss], addr_group=asize[miss] if addr[miss] else np.nan))
k = pd.DataFrame(rows)
k["either"] = k.by_name | k.by_addr

print("\n== missed matches an exact key reaches through a found sibling ==")
print(k.groupby("country")[["by_name", "by_addr", "either"]].mean().round(3).to_string())
for cap in (5, 10, 20, 50):
    ok = (k.by_name & (k.name_group <= cap)) | (k.by_addr & (k.addr_group <= cap))
    print(f"  only keys shared by <= {cap:>2} records: " +
          "  ".join(f"{c} {ok[k.country == c].mean():.3f}" for c in sorted(k.country.unique())))


# --- typo-tolerant keys: still exact hash lookups, but each one survives a different kind of noise
def prefixes(s, n=4):          # "consultants limitet" -> "cons limi"
    return " ".join(sorted({w[:n] for w in s.split()}))

def digits(s):                 # address numbers only
    return " ".join(sorted({w for w in s.split() if any(ch.isdigit() for ch in w)}))

KEYS = {
    "name_words": lambda i: key(name[i]),
    "addr_words": lambda i: key(addr[i]) if addr[i] else "",
    "name_prefix4": lambda i: prefixes(name[i]),
    "digits+name3": lambda i: (digits(addr[i]) + "|" + name[i][:3]) if digits(addr[i]) else "",
    "name2words+digits": lambda i: (" ".join(name[i].split()[:2]) + "|" + digits(addr[i])) if digits(addr[i]) else "",
}
CAP = 20
reach = {}
for kname, f in KEYS.items():
    k_all = pd.Series([f(i) for i in others], index=others)
    size = k_all.map(k_all.value_counts())
    hit = []
    for s1, miss in gt.loc[~gt.found, ["s1", "other"]].itertuples(index=False):
        km = k_all[miss]
        hit.append(km != "" and size[miss] <= CAP and any(k_all[s] == km for s in found_by_s1.get(s1, [])))
    reach[kname] = np.array(hit)

r = pd.DataFrame(reach)
r["any_key"] = r.any(axis=1)
r["country"] = [ctry[s] for s in gt.loc[~gt.found, "s1"]]
print(f"\n== missed matches reached through a found sibling, per key (groups <= {CAP}) ==")
print(r.groupby("country").mean().round(3).T.to_string())
for c in sorted(r.country.unique()):
    now = gt.found[gt_c == c].mean()
    extra = r.any_key[r.country == c].sum() / (gt_c == c).sum()
    print(f"{c:6} blocking recall now {now:.3f}  -> with these keys {now + extra:.3f}")
