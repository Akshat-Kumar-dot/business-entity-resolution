# recall of candidate_pairs.tsv against the ground truth, per country
import pandas as pd
from common import DATA, read_sources, read_tsv

cand = read_tsv("candidate_pairs.tsv")
gt = read_tsv(DATA / "train_ground_truth.tsv")
country = read_sources().set_index("entity_id").country


def pairs(df, col):
    p = df.assign(o=df[col].str.split(",")).explode("o")
    p = p[p.o.fillna("") != ""]
    return set(zip(p.source1_entity_id, p.o))


true, found = pairs(gt, "matched_entity_ids"), pairs(cand, "candidate_entity_ids")
hit = pd.DataFrame([(a, (a, b) in found) for a, b in true], columns=["s1", "hit"])
hit["country"] = hit.s1.map(country)
print(hit.groupby("country").hit.mean().round(4))
print("overall recall:", round(hit.hit.mean(), 4), "| candidates per s1:", round(len(found) / len(cand), 2))