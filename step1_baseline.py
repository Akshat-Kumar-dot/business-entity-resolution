# step 1: baseline. s2/s3 record is a candidate if same country and exact same name OR address
# python step1_baseline.py        raw text, only lowercase + punctuation removed
# python step1_baseline.py clean  normalized text (run normalize.py first), word order ignored
#
# we never build the pairs, only count group sizes -> cheap even on 12M rows

import sys

import pandas as pd

from common import CLEAN, load_truth, read_sources


def light(s):
    return s.str.lower().str.replace(r"[\W_]+", " ", regex=True).str.strip()


def sorted_words(s):
    # "pa camp hill borough" == "camp hill borough pa"
    return s.str.split().map(lambda w: " ".join(sorted(set(w))))


if "clean" in sys.argv:
    df = pd.read_parquet(CLEAN / "records.parquet")
    df["nk"], df["ak"] = sorted_words(df.name_norm), sorted_words(df.addr_norm)
else:
    df = read_sources()
    df["nk"], df["ak"] = light(df.business_name), light(df.business_address)

gt = load_truth(pd.Series(df.index, index=df.entity_id))
a, b = gt.s1.to_numpy(), gt.other.to_numpy()
nk, ak = df.nk.to_numpy(), df.ak.to_numpy()

# recall: is each true pair caught by name key, addr key, either
hit = pd.DataFrame({"country": df.country.to_numpy()[a],
                    "name": (nk[a] != "") & (nk[a] == nk[b]),
                    "addr": (ak[a] != "") & (ak[a] == ak[b])})
hit["either"] = hit.name | hit.addr
print("== recall ==")
print(hit.groupby("country").mean().round(4))
print("overall:", hit[["name", "addr", "either"]].mean().round(4).to_dict())


def n_cands(keys):
    # candidates of an s1 = size of its key group among s2/s3
    oth = df[df.source != "S1"]
    for k in keys:
        oth = oth[oth[k] != ""]   # empty key would match everything
    size = oth.groupby(["country", *keys]).size().rename("n")
    return df[df.source == "S1"].join(size, on=["country", *keys]).n.fillna(0)


c = df.loc[df.source == "S1", ["country"]]
c["name"], c["addr"] = n_cands(["nk"]), n_cands(["ak"])
c["union"] = c.name + c.addr - n_cands(["nk", "ak"])   # dont count "both" twice
print("\n== candidates per s1 ==")
print(c.groupby("country").agg(["mean", "max"]).round(2))
print("s1 with 0 candidates:", round((c.union == 0).mean(), 4))

print("\n== some missed pairs ==")
miss = gt[~hit.either.to_numpy()]
for i, j in miss.sample(min(12, len(miss)), random_state=1).itertuples(index=False):
    for r in (i, j):
        print(f"{df.entity_id[r]:>14} | {df.business_name[r]} | {df.business_address[r]}")
    print()
