# step 0: just look at the data
# python explore.py

import pandas as pd

from common import DATA, read_sources, read_tsv

df = read_sources()
print(pd.crosstab(df.source, df.country, margins=True))

gt = read_tsv(DATA / "train_ground_truth.tsv")
gt["other"] = gt.matched_entity_ids.str.split(",")
pairs = gt.explode("other")
pairs = pairs[pairs.other.fillna("") != ""]
print("\ntrue pairs:", len(pairs), pairs.other.str[:2].value_counts().to_dict())

n_matches = gt.other.apply(lambda ids: len([i for i in ids if i]))
print("\nmatches per s1:\n", n_matches.clip(upper=5).value_counts().sort_index())

country = df.set_index("entity_id").country
c1, c2 = pairs.source1_entity_id.map(country), pairs.other.map(country)
print("\nids missing from sources:", (c1.isna() | c2.isna()).sum(),
      "| cross country pairs:", (c1.notna() & c2.notna() & (c1 != c2)).sum())

# messy stuff
text = df.business_name + " " + df.business_address
for what, mask in {"empty address": df.business_address.str.strip() == "",
                   "non latin": text.str.contains(r"[^\x00-\x7F]"),
                   "junk chars": text.str.contains(r"#|\.com|www\.|[*_@]")}.items():
    print(f"\n[{what}] {mask.sum()}")
    print(df[mask].head(4).to_string(index=False))
