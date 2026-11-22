"""Final cleaning pass over data/clean/data.csv (in place).

Usage: python scripts/clean_dataset.py [data/clean/data.csv]

Run after extract_title_features.py and extract_description_features.py.

Steps
  rows     drop project/development pages, per-m2 prices, missing/implausible prices,
           implausible price-per-m2, duplicates
  areas    merge the many size columns into
             land_area_m2   plot size (houses/villas/shophouses/land)
             floor_area_m2  built area: unit size for condos, total over all floors for houses
             land_width_m / land_length_m  plot frontage x depth (house dims as fallback)
  types    property_type grouped into 8 classes
  location district / commune spellings unified across sites; lat/long dropped (39 rows only)
  lists    amenities / property_features split into 0/1 flag columns
  values   impossible counts and sizes set to missing
  flags    stored as 0/1; has_description = 1 when the listing had a description
           (khmer24 rows had none, so their text-derived flags are 0 for "unknown")
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "clean" / "data.csv"

TYPE_GROUP = {
    "Condo": "Condo", "Apartment": "Condo", "Studio": "Condo",
    "Serviced Apartment": "Condo", "Penthouse": "Condo",
    "Villa": "Villa", "Twin Villa": "Twin Villa", "Link House": "Link House",
    "Shophouse": "Shophouse", "Commercial Shophouse": "Shophouse",
    "Flat": "Flat",
    "House": "House", "House (unspecified)": "House", "Borey": "House", "Townhouse": "House",
    "Land": "Land",
}
UNIT = {"Condo"}

DISTRICT_FIX = {
    "Saensokh": "Sen Sok", "Meanchey": "Mean Chey", "Chrouy Changva": "Chroy Changvar",
    "Chamkarmon": "Chamkar Mon", "Toul Kork": "Tuol Kouk", "Por Sen Chey": "Por Senchey",
    "Ruessei Kaev": "Russey Keo", "Doun Penh": "Daun Penh", "Prampir Meakkakra": "7 Makara",
    "Preaek Pnov": "Prek Pnov", "Royal Palace": "Daun Penh", "Wat Phnom": "Daun Penh",
}
# token-level spelling variants for communes (applied to lower-cased names)
COMMUNE_TOKENS = [
    (r"\bbkk\b", "boeng keng kang"), (r"\bboeung\b", "boeng"), (r"\btoul\b", "tuol"),
    (r"\btum poung\b", "tumpung"), (r"\bbasak\b", "bassac"), (r"\bthmey\b", "thmei"),
    (r"\bchamres\b", "chamreh"), (r"\bkakap\b", "kakab"), (r"\bsnaor\b", "snao"),
    (r"\bmittakpheap\b", "mittapheap"), (r"\bkiloumaetr\b", "kilomaetr"), (r"\bteuk\b", "tuek"),
    (r"\bprek\b", "preaek"), (r"\bliep\b", "lieb"), (r"\bchrouy changva\b", "chroy changvar"),
    (r"\bruessei kaev\b", "russey keo"), (r"\bvoat phnum\b", "wat phnom"), (r"\bboeng trabek\b", "boeng trabaek"),
    (r"\bti\b", ""),
    (r"\bmuoy\b|\bi\b", "1"), (r"\bpir\b|\bii\b", "2"), (r"\bbei\b|\biii\b", "3"), (r"\bbuon\b|\biv\b", "4"),
]

AMENITY_FLAGS = {
    "Car Parking": "has_parking", "Swimming Pool": "has_pool", "Garden": "has_garden",
    "On main road": "on_main_road", "Non-Flooding": "non_flooding", "Commercial area": "commercial_area",
    "Common Area": "common_area", "Lift / Elevator": "has_lift", "Gym/Fitness Center": "has_gym",
    "Backup Electricity / Generator": "has_generator", "Playground": "has_playground",
    "Sports Facilities": "has_sports", "Sauna": "has_sauna", "Jacuzzi": "has_jacuzzi",
    "Tennis Court": "has_tennis",
    # property_features
    "Fully Furnished": "is_furnished", "Balcony": "has_balcony", "Air Conditioning": "has_aircon",
    "Internet / Wifi": "has_internet", "Serviced Property": "is_serviced",
    "Pet Friendly": "pet_friendly", "Pay TV": "has_pay_tv",
}

# (column, min, max): values outside become missing
BOUNDS = [
    ("bedrooms", 0, 20), ("bathrooms", 0, 20), ("floors", 1, 10), ("floor_level", 0, 70),
    ("living_rooms", 0, 10), ("kitchens", 0, 10), ("front_space_m", 0, 50), ("back_space_m", 0, 50),
    ("land_area_m2", 10, 1_000_000), ("floor_area_m2", 10, 20_000),
    ("land_width_m", 1, 500), ("land_length_m", 1, 1000),
]

ORDER = [
    "price_usd", "property_type", "province", "district", "commune", "borey_name", "in_borey",
    "title_deed", "land_area_m2", "floor_area_m2", "land_width_m", "land_length_m",
    "front_space_m", "back_space_m", "floors", "floor_level", "bedrooms", "bathrooms",
    "living_rooms", "kitchens", "has_description",
]


def norm_commune(name):
    if pd.isna(name):
        return name
    s = " " + str(name).lower().strip() + " "
    for pat, rep in COMMUNE_TOKENS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w if w.isdigit() else w.capitalize() for w in s.split())


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "area_type" not in d:
        sys.exit(f"{path}: already cleaned")
    n0 = len(d)
    log = []

    def drop(mask, why):
        nonlocal d
        log.append(f"drop {int(mask.sum()):>5}  {why}")
        d = d[~mask].copy()

    # ---- provenance indicator before raw columns go
    d["has_description"] = d["area_type"].isna()           # khmer24 rows always have area_type
    camb = d["area_type"].isna() & d["area_m2"].notna()     # cambodiaproperty: ambiguous "Area Size"
    notes = d["notes"].fillna("")
    d["is_negotiable"] = d["is_negotiable"].fillna(False).astype(bool) | notes.str.contains("negotiable")

    # ---- rows
    drop(d["is_project"].astype(bool) | notes.str.contains("project") | d["property_type"].eq("Project"),
         "project/development pages")
    drop(d["price_is_per_m2"].astype(bool), "price quoted per m2")
    drop(d["price_usd"].isna() | (d["price_usd"] < 5_000) | (d["price_usd"] > 20_000_000),
         "price missing, < $5k or > $20M")

    # ---- property type
    d["property_type"] = d["property_type"].map(TYPE_GROUP)
    drop(d["property_type"].isna(), "unknown property type")
    unit = d["property_type"].isin(UNIT)

    # ---- areas
    camb = camb.reindex(d.index, fill_value=False)
    d.loc[camb & unit, "floor_area_m2"] = d.loc[camb & unit, "floor_area_m2"].fillna(d["area_m2"])
    d.loc[camb & ~unit, "land_area_m2"] = d.loc[camb & ~unit, "land_area_m2"].fillna(d["area_m2"])
    for a, b in (("land_width_m", "width_m"), ("land_length_m", "length_m")):
        d.loc[camb & ~unit, a] = d.loc[camb & ~unit, a].fillna(d[b])
    d["land_area_m2"] = d["land_area_m2"].fillna(d["land_width_m"] * d["land_length_m"])
    # houses: built area = footprint x floors when not given
    footprint = d["house_width_m"] * d["house_length_m"]
    d.loc[~unit, "floor_area_m2"] = d.loc[~unit, "floor_area_m2"].fillna(footprint * d["floors"])
    # frontage/depth: plot dims, else house dims (shophouses/flats usually span the plot width)
    d["land_width_m"] = d["land_width_m"].fillna(d["house_width_m"])
    d["land_length_m"] = d["land_length_m"].fillna(d["house_length_m"])
    # condos have no plot
    d.loc[unit, ["land_area_m2", "land_width_m", "land_length_m", "front_space_m", "back_space_m", "floors"]] = np.nan
    d.loc[d["property_type"].eq("Land"), ["floor_area_m2", "floors", "bedrooms", "bathrooms"]] = np.nan

    # ---- value bounds
    for col, lo, hi in BOUNDS:
        bad = d[col].notna() & ((d[col] < lo) | (d[col] > hi))
        if bad.any():
            log.append(f"null  {int(bad.sum()):>5}  {col} outside [{lo}, {hi}]")
        d.loc[bad, col] = np.nan
    big_unit = unit & (d["floor_area_m2"] > 1000)
    d.loc[big_unit, "floor_area_m2"] = np.nan
    log.append(f"null  {int(big_unit.sum()):>5}  condo floor_area_m2 > 1000")

    # ---- implausible price per m2 (bad price or bad area)
    ppm_unit = d["price_usd"] / d["floor_area_m2"].where(unit)
    ppm_land = d["price_usd"] / d["land_area_m2"].where(~unit)
    drop((ppm_unit < 200) | (ppm_unit > 15_000), "condo price/m2 outside $200-$15k")
    ppm_land = ppm_land.reindex(d.index)
    drop((ppm_land < 5) | (ppm_land > 60_000), "price/land m2 outside $5-$60k")

    # ---- location
    d["district"] = d["district"].replace(DISTRICT_FIX)
    d["commune"] = d["commune"].map(norm_commune)
    d["borey_name"] = d["borey_name"].str.strip().str.replace(r"\s+", " ", regex=True)
    d["in_borey"] = d["borey_name"].notna()
    d["title_deed"] = d["title_deed"].fillna("Unknown")

    # ---- amenity / feature lists -> flags
    items = (d["amenities"].fillna("") + "; " + d["property_features"].fillna("")).str.split(r";\s*")
    items = items.map(set)
    for label, col in AMENITY_FLAGS.items():
        hit = items.map(lambda s, l=label: l in s)
        d[col] = (d[col].fillna(False).astype(bool) | hit) if col in d else hit

    # ---- drop merged / raw / helper columns
    d = d.drop(columns=[
        "latitude", "longitude", "property_type_raw", "area_m2", "house_area_m2",
        "house_width_m", "house_length_m", "width_m", "length_m", "area_raw", "area_type",
        "land_size_raw", "building_size_raw", "land_dim_raw", "house_dim_raw", "dim_raw",
        "property_features", "amenities", "notes", "is_project", "price_is_per_m2",
    ])

    # ---- duplicates (exact, then same listing cross-posted with identical key fields)
    drop(d.duplicated(keep="first"), "exact duplicates")
    key = ["price_usd", "property_type", "district", "commune", "bedrooms", "bathrooms",
           "land_area_m2", "floor_area_m2", "floor_level", "borey_name"]
    drop(d.duplicated(subset=key, keep="first"), "duplicates on key fields")

    # ---- dtypes / order
    flags = [c for c in d.columns if d[c].dtype == bool]
    d[flags] = d[flags].astype(int)
    d = d[ORDER + sorted(c for c in d.columns if c not in ORDER)]
    d.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n".join(log))
    print(f"rows {n0} -> {len(d)}, cols {d.shape[1]}")


if __name__ == "__main__":
    main()
