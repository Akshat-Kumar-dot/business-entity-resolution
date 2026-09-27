# step 2: clean names + addresses once and save them -> clean/records.parquet
# pip install anyascii pyarrow
# python normalize.py
#
# only generic rules + short word lists, nothing tied to US/India formats (test has France)

import pandas as pd
from anyascii import anyascii

from common import CLEAN, read_sources

# never real words -> drop anywhere in the name
LEGAL_ANY = "llc llp pllc inc corp pvt ltd plc opc sarl sasu eurl".split()
# can be real words (co, sa, company..) -> only drop at the end
LEGAL_END = "co company corporation incorporated limited private pc pa lp sa sas sci snc".split()

ABBR = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "drive": "dr", "lane": "ln", "trail": "trl",
    "highway": "hwy", "suite": "ste", "floor": "fl", "flr": "fl", "building": "bldg",
    "near": "nr", "opposite": "opp", "place": "pl", "chemin": "ch", "route": "rte",
}


def any_word(ws):
    return r"\b(?:" + "|".join(ws) + r")\b"


def clean(s):
    # shared by names and addresses, order matters
    s = s.astype(object).str.normalize("NFKC")   # object dtype: pandas' arrow regex can't do \1
    odd = s.str.contains(r"[^\x00-\x7f]")
    s[odd] = s[odd].map(anyascii)                # other scripts + accents -> latin (slow, so only where needed)
    s = s.str.lower()
    s = s.str.replace(r"(?<!\w)(?:n/a|null|none|nil)(?!\w)", " ", regex=True)
    s = s.str.replace(r"www\.|\.(?:com|co\.in|in|net|org|fr)\b", " ", regex=True)
    s = s.str.replace(r"(?<=\b[a-z])\.", "", regex=True)          # p.a. -> pa, l.l.c. -> llc
    s = s.str.replace(r"['’]", "", regex=True)                     # orelee's -> orelees
    s = s.str.replace(r"[\W_]+", " ", regex=True)                  # rest of punctuation -> space
    s = s.str.replace(r"\b0+(\d)", r"\1", regex=True)              # 01715 -> 1715
    s = s.str.replace(r"\b(\w+)(?: \1\b)+", r"\1", regex=True)     # llc llc -> llc
    return s.str.strip()


def clean_name(s):
    s = clean(s).str.replace(r"^(?:m s|messrs) ", "", regex=True)  # "M/s Jg Bhoomi"
    out = s.str.replace(any_word(LEGAL_ANY), " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.str.replace(r"(?: (?:" + "|".join(LEGAL_END) + r"))+$", "", regex=True)
    return out.where(out != "", s)   # dont wipe out a name that is only "sa co"


def clean_addr(s):
    return clean(s).str.replace(any_word(ABBR), lambda m: ABBR[m.group(0)], regex=True)


if __name__ == "__main__":
    print("reading..", flush=True)
    df = read_sources()
    print("names..", flush=True)
    df["name_norm"] = clean_name(df.business_name)
    print("addresses..", flush=True)
    df["addr_norm"] = clean_addr(df.business_address)

    # house no = first token with a digit, postcode = last 5-6 digit token (us zip / in pin / fr)
    df["house_no"] = df.addr_norm.str.extract(r"\b(\w*\d\w*)\b", expand=False).fillna("")
    pc = df.addr_norm.str.extract(r".*\b(\d{5,6})\b", expand=False)
    df["postcode"] = pc.where(pc != df.house_no).fillna("")

    CLEAN.mkdir(exist_ok=True)
    df.to_parquet(CLEAN / "records.parquet", index=False)
    print("saved", CLEAN / "records.parquet")

    print("\nshare of records that have it:")
    print((df[["name_norm", "addr_norm", "house_no", "postcode"]] != "").groupby(df.country).mean().round(3))

    # eyeball some before/after
    raw = df.business_name + " " + df.business_address
    for r in df[raw.str.contains(r"[^\x00-\x7f]|#|\.com|\b(?i:llc|pvt|ltd)\b")].head(8).itertuples():
        print(f"\n  {r.business_name} | {r.business_address}\n  -> {r.name_norm} | {r.addr_norm}")
