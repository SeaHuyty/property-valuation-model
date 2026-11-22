"""Mine listing descriptions for features, then drop description (and updated_raw).

Usage: python scripts/extract_description_features.py [data/clean/data.csv] [--check]

  --check   only report how often parsed values agree with existing structured
            values; write nothing.

Edits the CSV in place. Only missing values are filled; structured values win.
Filled from description:
  bedrooms, bathrooms, floors ("3 floors", "E0, E1, E2", "3-storey"),
  floor_level ("20th floor"), title_deed, floor_area_m2 ("Net/Gross Area: 132 sqm",
  "size 65.8 m2" for units), land_width_m/land_length_m/land_area_m2 ("Land size 5m x 21.5m"),
  house_width_m/house_length_m ("House size 5m x 12m")
New columns:
  front_space_m, back_space_m   open yard in front/back of the house
  living_rooms, kitchens
  is_negotiable, has_installment, on_main_road, has_pool, has_parking, has_garden
Flags is_corner / is_furnished / has_view / is_duplex are OR-ed with description hits.
"""
import re
import sys
from pathlib import Path

import pandas as pd

from extract_title_features import BATH_RE, BED_RE, first_num

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "clean" / "data.csv"

NUM = r"(\d+(?:\.\d+)?)"
DIM = rf"{NUM}\s*m?\s*[x×*]\s*{NUM}\s*m?"
TAG = r"(?:<[^>]+>|\*\*|\s|:|-)*"  # html bold / markdown / punctuation between label and value

LAND_RE = re.compile(rf"land\s*(?:size|area|dimension)?{TAG}{DIM}", re.I)
HOUSE_RE = re.compile(rf"(?:house|building)\s*(?:size|area|dimension)?{TAG}{DIM}", re.I)
LAND_M2_RE = re.compile(rf"land\s*(?:size|area){TAG}{NUM}\s*(?:m2|m²|sqm)", re.I)
UNIT_M2_RE = re.compile(rf"(?:net|gross|unit|floor|total|living)?\s*(?:area|size){TAG}{NUM}\s*(?:m2|m²|sqm|sq\.?\s?m)", re.I)
# Sites store gross area; descriptions often give both ("42 sqm net (54 sqm gross)").
GROSS_RE = re.compile(rf"gross\s*(?:area|size)?{TAG}{NUM}\s*(?:m2|m²|sqm|sq\.?\s?m)|{NUM}\s*(?:m2|m²|sqm|sq\.?\s?m)\s*\(?\s*gross", re.I)
FLOORS_RE = re.compile(r"(?<![\d.])(\d)\s*-?\s*(?:floors?|stor(?:e)?y|stories|storeys)\b(?!\s*(?:apartment|unit|condo))", re.I)
E_FLOORS_RE = re.compile(r"\bE0(?:\s*,\s*E\d)+", re.I)
LEVEL_RE = re.compile(r"(\d{1,2})\s*(?:st|nd|rd|th)\s*floor|floor\s*(?:level)?\s*[:\-]?\s*(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
FRONT_RE = re.compile(rf"front\s*(?:yard|space|open space)?{TAG}{NUM}\s*m\b|{NUM}\s*m\s*(?:in\s*)?(?:the\s*)?front", re.I)
BACK_RE = re.compile(rf"back\s*(?:yard|space)?{TAG}{NUM}\s*m\b|{NUM}\s*m\s*(?:in\s*)?(?:the\s*)?back", re.I)
LIVING_RE = re.compile(r"(\d)\s*living\s*rooms?", re.I)
KITCHEN_RE = re.compile(r"(\d)\s*kitchens?", re.I)
TITLE_RES = [
    (re.compile(r"strata\s*title", re.I), "Strata Title"),
    (re.compile(r"soft\s*title", re.I), "Soft Title"),
    (re.compile(r"hard\s*title|ប្លង់រឹង", re.I), "Hard Title"),
    (re.compile(r"leasehold", re.I), "Long-Term Leasehold"),
]
UNIT_TYPES = {"Condo", "Apartment", "Studio", "Serviced Apartment", "Penthouse"}


def grp(m, i=None):
    if not m:
        return None
    vals = [g for g in m.groups() if g]
    return float(vals[0]) if i is None else float(vals[i])


def parse(desc: str) -> dict:
    out = {}
    if not desc:
        return out
    if m := LAND_RE.search(desc):
        out["land_width_m"], out["land_length_m"] = float(m.group(1)), float(m.group(2))
        out["land_area_m2"] = round(out["land_width_m"] * out["land_length_m"], 2)
    elif m := LAND_M2_RE.search(desc):
        out["land_area_m2"] = float(m.group(1))
    if m := HOUSE_RE.search(desc):
        out["house_width_m"], out["house_length_m"] = float(m.group(1)), float(m.group(2))
    if m := GROSS_RE.search(desc):
        out["unit_m2"] = grp(m)
    elif m := UNIT_M2_RE.search(desc):
        out["unit_m2"] = float(m.group(1))
    if m := E_FLOORS_RE.search(desc):
        out["floors"] = len(re.findall(r"E\d", m.group(0)))
    elif m := FLOORS_RE.search(desc):
        out["floors"] = int(m.group(1))
    if m := LEVEL_RE.search(desc):
        out["floor_level"] = grp(m)
    out["bedrooms"] = first_num(BED_RE.search(desc))
    out["bathrooms"] = first_num(BATH_RE.search(desc))
    out["front_space_m"] = grp(FRONT_RE.search(desc))
    out["back_space_m"] = grp(BACK_RE.search(desc))
    out["living_rooms"] = grp(LIVING_RE.search(desc))
    out["kitchens"] = grp(KITCHEN_RE.search(desc))
    for rx, name in TITLE_RES:
        if rx.search(desc):
            out["title_deed"] = name
            break
    return out


FLAGS = {
    "is_negotiable": r"negotiab|ចរចា",
    "has_installment": r"installment|instalment|payment plan|monthly payment|បង់រំលស់",
    "on_main_road": r"main road|main street|along (?:the )?(?:\w+ )?(?:road|blvd|boulevard)|ផ្លូវធំ|ផ្លូវមេ",
    "has_pool": r"swimming pool|\bpool\b",
    "has_parking": r"parking|car park|garage",
    "has_garden": r"garden",
}
OR_FLAGS = {
    "is_corner": r"corner|កែង",
    "is_furnished": r"(?<!un)(?<!un-)furnish",
    "has_view": r"\bview\b",
    "is_duplex": r"duplex",
}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    path = Path(args[0]) if args else DEFAULT
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "description" not in d:
        sys.exit(f"{path}: no description column (already processed?)")
    desc = d["description"].fillna("").str.replace(r"<[^>]+>", "", regex=True)
    p = pd.DataFrame(desc.map(parse).tolist(), index=d.index)
    unit = d["property_type"].isin(UNIT_TYPES)
    p["floor_area_m2"] = p["unit_m2"].where(unit)
    p["floor_level"] = p["floor_level"].where(unit | d["property_type"].eq("Flat"))
    p["floors"] = p["floors"].where(~unit)  # condo text gives the tower's storeys, not the unit's

    fill_cols = ["bedrooms", "bathrooms", "floors", "floor_level", "title_deed", "floor_area_m2",
                 "land_width_m", "land_length_m", "land_area_m2", "house_width_m", "house_length_m"]
    print(f"{'column':<16}{'parsed':>8}{'both':>7}{'agree':>8}{'fills':>7}")
    stats = {}
    for col in fill_cols:
        if col not in p:
            continue
        v = p[col]
        both = v.notna() & d[col].notna()
        if col == "title_deed":
            agree = (v[both] == d.loc[both, col]).mean()
        else:
            agree = (abs(pd.to_numeric(v[both]) - d.loc[both, col]) <= 0.05 * d.loc[both, col].abs() + 0.01).mean()
        fills = d[col].isna() & v.notna()
        stats[col] = fills
        print(f"{col:<16}{int(v.notna().sum()):>8}{int(both.sum()):>7}{agree:>8.1%}{int(fills.sum()):>7}")
    if check:
        return

    for col, fills in stats.items():
        d.loc[fills, col] = p.loc[fills, col]
    for col in ["front_space_m", "back_space_m", "living_rooms", "kitchens"]:
        d[col] = p[col]
        print(f"{col}: {int(d[col].notna().sum())}")
    for col, pat in FLAGS.items():
        d[col] = desc.str.contains(pat, case=False, regex=True)
        print(f"{col}: {int(d[col].sum())}")
    for col, pat in OR_FLAGS.items():
        before = int(d[col].sum())
        d[col] = d[col].astype(bool) | desc.str.contains(pat, case=False, regex=True)
        print(f"{col}: {before} -> {int(d[col].sum())}")

    d = d.drop(columns=[c for c in ("description", "updated_raw") if c in d])
    d.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(d)} rows x {d.shape[1]} cols -> {path}")


if __name__ == "__main__":
    main()
