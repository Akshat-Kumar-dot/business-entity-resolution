# stuff every step needs
#
# train or test? set it before running any step (powershell):
#   $env:SPLIT="test"      then run the steps as usual
#   $env:SPLIT="train"     back to train (default)

import csv
import os
from pathlib import Path

import pandas as pd

SPLIT = os.environ.get("SPLIT", "train")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))   # stops the long "wmic" warning on windows
ROOT = Path(__file__).parent
DATA = ROOT / "student_resource" / "dataset" / SPLIT
CLEAN = ROOT / ("clean" if SPLIT == "train" else f"clean_{SPLIT}")   # test never overwrites train files
TRAINED = ROOT / "clean"          # models are always the ones trained on train
SUBMIT = ROOT / "output"          # final files for the test set go here (the zip wants output/)


def read_tsv(path):
    # keep "NA" / "" as text, and dont let a stray " swallow the next rows
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def read_sources():
    # train_source1.tsv / test_source1.tsv ... whatever the prefix is
    files = sorted(DATA.glob("*source[123].tsv"))
    if len(files) != 3:
        raise SystemExit(f"expected 3 source files in {DATA}, found {[f.name for f in files]}")
    df = pd.concat([read_tsv(f) for f in files], ignore_index=True)
    df["source"] = df.entity_id.str[:2]
    return df


def load_truth(row_of):
    # ground truth is one row per s1 with "S2-..,S3-.." -> one (s1 row, other row) per match
    path = next(DATA.glob("*ground_truth.tsv"), None)
    if path is None:   # test set: no answers, return an empty table
        return pd.DataFrame({"s1": pd.Series(dtype=int), "other": pd.Series(dtype=int)})
    gt = read_tsv(path)
    gt["other"] = gt.matched_entity_ids.str.split(",")
    gt = gt.explode("other")
    gt = gt[gt.other.fillna("") != ""]
    return pd.DataFrame({"s1": gt.source1_entity_id.map(row_of).to_numpy(),
                         "other": gt.other.map(row_of).to_numpy()})


def load_clean():
    return pd.read_parquet(CLEAN / "records.parquet",
                           columns=["entity_id", "source", "country", "name_norm", "addr_norm"])


def pick_s1(df, sample):
    # same random s1 subset in every step, so --sample runs line up
    s1 = df[df.source == "S1"]
    return s1.sample(sample, random_state=0) if sample else s1
