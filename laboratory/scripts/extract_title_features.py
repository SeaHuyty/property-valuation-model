"""Mine listing titles for features, then drop the title column.

Usage: python scripts/extract_title_features.py [data/clean/data.csv]

Edits the CSV in place. Titles hold info that the structured fields miss:
  - bedrooms / bathrooms  ("4 Bed, 5 Bath", "2Bedroom", "One Bedroom", "បន្ទប់គេង 3")
  - floor_area_m2         ("75sqm", "127m2") for condo-type units
  - floor_level           ("37th Floor")
  - borey_name            ("... in Borey Peng Huoth: The Star ...")
  - title_deed            ("hard title", "ប្លង់រឹង")
  - property_type         realestate.com.kh mislabels: Retreat = Flat House, Unit = Link House;
                          khmer24 "House (unspecified)" resolved from Khmer/English type words
  - listing_type          rentals posted as sales ("2 bed rent $750")
New columns:
  price_is_per_m2   title quotes "$/m2" and price is a per-m2 figure, not the total
  is_corner, is_furnished, has_view, is_duplex, is_urgent
Only missing values are filled; existing structured values are never overwritten
(except the property_type relabels above).
"""
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "clean" / "data.csv"

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "មួយ": 1, "ពីរ": 2, "បី": 3, "បួន": 4, "ប្រាំ": 5}
# "5+1" = 5 bedrooms + 1 maid/guest room -> 6
NUMW = r"(\d{1,2}(?:\s*\+\s*\d)?|one|two|three|four|five|six|មួយ|ពីរ|បី|បួន|ប្រាំ)"
BED_RE = re.compile(rf"{NUMW}\s*-?\s*(?:bed(?:room)?s?|br)\b|{NUMW}\s*b\+|{NUMW}\s*បន្ទប់គេង|បន្ទប់គេង(?:សរុប)?\s*(\d{{1,2}})", re.I)
BATH_RE = re.compile(rf"{NUMW}\s*-?\s*(?:bath(?:room)?s?)\b|{NUMW}\s*បន្ទប់ទឹក|បន្ទប់ទឹក(?:សរុប)?\s*(\d{{1,2}})", re.I)
SIZE_RE = re.compile(r"(\d{2,4}(?:\.\d+)?)\s*(?:m2|m²|sqm|sq\.?\s?m|ម៉ែត្រការ៉េ)(?!\s*/)", re.I)
FLOOR_RE = re.compile(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*floors?\b", re.I)
BOREY_RE = re.compile(r"\bin (Borey [^|,]+?)\s*$")
# "បន្ទប់ជួល" = rental rooms inside a house that is for sale, not a rent offer
RENT_RE = re.compile(r"\brent\b|\blease\b|(?<!បន្ទប់)ជួល", re.I)
PER_M2_RE = re.compile(r"\$\s*/\s*(?:m2|m²|sqm)|\$/m|/\s*(?:m2|m²|sqm)\b|ក្នុង\s*1\s*ម៉ែត្រ", re.I)

# Order matters: more specific first.
KH_TYPES = [
    (r"វីឡាភ្លោះ|twin\s*villa", "Twin Villa"),
    (r"ផ្ទះអាជីវកម្ម|shop\s*house|shophouse", "Shophouse"),
    (r"ផ្ទះល្វែង|\bflat\b", "Flat"),
    (r"\bLA\b|link\s*house", "Link House"),
    (r"វីឡា|villa", "Villa"),
    (r"^ដី|\bland\b", "Land"),
]
UNIT_TYPES = {"Condo", "Apartment", "Studio", "Serviced Apartment", "Penthouse", "Flat"}


def first_num(m):
    if not m:
        return None
    g = next(x for x in m.groups() if x)
    if g.lower() in WORDS:
        return WORDS[g.lower()]
    return sum(int(n) for n in g.split("+"))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "title" not in d:
        sys.exit(f"{path}: no title column (already processed?)")
    t = d["title"].fillna("")
    stats = {}

    def fill(col, values):
        mask = d[col].isna() & values.notna()
        d.loc[mask, col] = values[mask]
        stats[col] = int(mask.sum())

    # counts
    fill("bedrooms", t.map(lambda s: first_num(BED_RE.search(s))))
    studio = t.str.contains(r"studio", case=False)
    fill("bedrooms", pd.Series(1.0, index=d.index).where(studio))  # site convention: studio = 1
    stats["bedrooms"] += int((studio & d["bedrooms"].isna()).sum())
    fill("bathrooms", t.map(lambda s: first_num(BATH_RE.search(s))))

    # size, floor, borey, title deed
    unit = d["property_type"].isin(UNIT_TYPES)
    size = t.map(lambda s: (m := SIZE_RE.search(s)) and float(m.group(1)))
    fill("floor_area_m2", pd.to_numeric(size, errors="coerce").where(unit))
    fill("floor_level", pd.to_numeric(t.map(lambda s: (m := FLOOR_RE.search(s)) and int(m.group(1))), errors="coerce").where(unit))
    fill("borey_name", t.map(lambda s: (m := BOREY_RE.search(s)) and m.group(1).strip()))
    fill("title_deed", pd.Series("Hard Title", index=d.index).where(t.str.contains(r"hard title|ប្លង់រឹង", case=False)))

    # property_type fixes
    relabel = {"Retreat": "Flat", "Unit": "Link House"}
    stats["property_type relabel"] = int(d["property_type"].isin(relabel).sum())
    d["property_type"] = d["property_type"].replace(relabel)
    unspec = d["property_type"].eq("House (unspecified)")
    resolved = pd.Series(pd.NA, index=d.index, dtype="object")
    for pat, name in reversed(KH_TYPES):  # reversed so specific patterns win
        resolved[t.str.contains(pat, case=False, regex=True)] = name
    m = unspec & resolved.notna()
    d.loc[m, "property_type"] = resolved[m]
    stats["property_type resolved"] = int(m.sum())

    # pricing signals
    price = d["price_usd"]
    rent = t.str.contains(RENT_RE) & price.lt(10_000)
    stats["listing_type -> rent"] = int((rent & d["listing_type"].ne("rent")).sum())
    d.loc[rent, "listing_type"] = "rent"
    sale_and_rent = t.str.contains(RENT_RE) & price.ge(10_000) & d["listing_type"].eq("sale")
    d.loc[sale_and_rent, "listing_type"] = "both"
    stats["listing_type -> both"] = int(sale_and_rent.sum())
    d["price_is_per_m2"] = t.str.contains(PER_M2_RE) & price.lt(20_000)

    # flags
    flags = {
        "is_corner": r"corner|កែង",
        "is_furnished": r"furnish|សម្ភារ",
        "has_view": r"\bview\b|ទេសភាព",
        "is_duplex": r"duplex",
        "is_urgent": r"urgent|បន្ទាន់",
    }
    for col, pat in flags.items():
        d[col] = t.str.contains(pat, case=False, regex=True)
    feats = d["property_features"].fillna("")
    d["is_furnished"] |= feats.str.contains("Furnished", case=False)

    d = d.drop(columns="title")
    d.to_csv(path, index=False, encoding="utf-8-sig")
    for k, v in stats.items():
        print(f"{k}: {v}")
    for col in ["price_is_per_m2", *flags]:
        print(f"{col}: {int(d[col].sum())}")
    print(f"wrote {len(d)} rows x {d.shape[1]} cols -> {path}")


if __name__ == "__main__":
    main()
