"""Consolidate the per-site scrapes into one dataset.

Usage: python scripts/consolidate_listings.py [--out data/listings_consolidated.csv]

Inputs (data/):
  cambodiaproperty.csv          cambodiaproperty.asia
  khmer24_listings.csv          khmer24.com
  realestate_kh.csv             realestate.com.kh (houses)
  realestate_kh_apartment.csv   realestate.com.kh (apartments/condos)

Every column from every source is kept (union schema). Columns that mean the
same thing across sites share one name; site-specific columns are blank for the
other sources. A few derived columns are added:
  source, source_file      origin of the row
  listing_id               filled from the URL when the scraper left it blank
  property_type            normalised casing; original kept in property_type_raw
  area_m2                  parsed from khmer24 area_raw (cambodiaproperty: as scraped,
                           the site's ambiguous "Area Size", usually land)
  land_area_m2, land_width_m, land_length_m     parsed from khmer24 land_size_raw
  house_area_m2, house_width_m, house_length_m  parsed from khmer24 building_size_raw
  is_project               notes flag the row as a project/development page
"""
import argparse
import csv
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OUT = DATA / "listings_consolidated.csv"

NUM = r"(\d+(?:[.,]\d+)?)"
DIM_RE = re.compile(rf"{NUM}\s*(?:m|meters?)?\s*[x×*]\s*{NUM}", re.I)
AREA_RE = re.compile(NUM)

# Output column order: shared fields first, then site-specific ones.
COLUMNS = [
    "source", "source_file", "url", "listing_id", "title", "listing_type",
    "listed_raw", "updated_raw",
    "province", "district", "commune", "latitude", "longitude",
    "property_type", "property_type_raw", "borey_name", "title_deed", "facing",
    "price_usd", "bedrooms", "bathrooms", "car_spaces", "floors", "floor_level",
    # areas
    "area_m2", "land_area_m2", "floor_area_m2", "house_area_m2",
    "land_width_m", "land_length_m", "house_width_m", "house_length_m",
    "width_m", "length_m",
    # raw size strings
    "area_raw", "area_type", "land_size_raw", "building_size_raw",
    "land_dim_raw", "house_dim_raw", "dim_raw",
    # text
    "property_features", "amenities", "description", "notes", "is_project",
]

TYPE_FIX = {
    "condo": "Condo", "project": "Project", "borey": "Borey",
    "ServicedApartment": "Serviced Apartment", "CommercialShophouse": "Commercial Shophouse",
}


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def parse_dims(value):
    """'4.2m x 16m' -> (4.2, 16.0); None when not a W x L string."""
    if pd.isna(value):
        return None, None
    m = DIM_RE.search(str(value))
    return (_f(m.group(1)), _f(m.group(2))) if m else (None, None)


def parse_area(value):
    """m2 from 'W x L' or a single total-area value ('120', '120 m2', '122m²')."""
    if pd.isna(value) or not str(value).strip():
        return None
    w, l = parse_dims(value)
    if w is not None:
        return round(w * l, 2)
    m = AREA_RE.search(str(value))
    return _f(m.group(1)) if m else None


def read_realestate(path: Path) -> pd.DataFrame:
    """Read a realestate.com.kh CSV, repairing rows where an interrupted append
    glued two records together (the first record lost its trailing `notes`)."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    n = len(header)
    fixed = []
    for r in body:
        if len(r) == n:
            fixed.append(r)
        elif len(r) == 2 * n - 1 and r[n - 1].startswith("http"):
            fixed.append(r[: n - 1] + [""])
            fixed.append(r[n - 1:])
            print(f"  repaired merged row in {path.name}: {r[0]} + {r[n - 1]}")
        else:
            raise ValueError(f"{path.name}: unexpected row with {len(r)} fields: {r[:1]}")
    df = pd.DataFrame(fixed, columns=header).replace("", pd.NA)
    df["source"] = "realestate.com.kh"
    return df


def read_cambodiaproperty(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns={"features": "property_features"})
    df["source"] = "cambodiaproperty.asia"
    df["listing_type"] = "sale"
    return df


def read_khmer24(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df["source"] = "khmer24.com"
    df["listing_type"] = "sale"
    df["listing_id"] = df["url"].str.extract(r"adid-(\d+)", expand=False)
    df["area_m2"] = df["area_raw"].map(parse_area)
    df["land_area_m2"] = df["land_size_raw"].map(parse_area)
    df["house_area_m2"] = df["building_size_raw"].map(parse_area)
    df[["land_width_m", "land_length_m"]] = df["land_size_raw"].map(parse_dims).tolist()
    df[["house_width_m", "house_length_m"]] = df["building_size_raw"].map(parse_dims).tolist()
    # Condo listings: the site's size is the unit's floor area.
    building = df["area_type"].eq("building")
    df.loc[building, "floor_area_m2"] = df.loc[building, "area_m2"]
    return df


SOURCES = [
    ("cambodiaproperty.csv", read_cambodiaproperty),
    ("khmer24_listings.csv", read_khmer24),
    ("realestate_kh.csv", read_realestate),
    ("realestate_kh_apartment.csv", read_realestate),
]

NUMERIC = [
    "latitude", "longitude", "price_usd", "bedrooms", "bathrooms", "car_spaces",
    "floors", "floor_level", "area_m2", "land_area_m2", "floor_area_m2", "house_area_m2",
    "land_width_m", "land_length_m", "house_width_m", "house_length_m", "width_m", "length_m",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    frames = []
    for name, reader in SOURCES:
        df = reader(DATA / name)
        df["source_file"] = name
        print(f"{name}: {len(df)} rows, {df.shape[1]} cols")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    missing = set(df.columns) - set(COLUMNS)
    assert not missing, f"columns dropped by COLUMNS list: {missing}"

    # listing_id from URL where missing (cambodiaproperty project pages etc.)
    from_url = df["url"].str.extract(r"-(\d+)/?$", expand=False)
    df["listing_id"] = df["listing_id"].fillna(from_url)

    df["property_type_raw"] = df["property_type"]
    df["property_type"] = df["property_type"].replace(TYPE_FIX)

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_project"] = df["notes"].fillna("").str.contains("project", case=False)

    before = len(df)
    df = df.drop_duplicates(subset="url", keep="first")
    print(f"dropped {before - len(df)} duplicate URLs")

    df = df.reindex(columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"wrote {len(df)} rows x {df.shape[1]} cols -> {args.out}")
    print(df.groupby("source").size().to_string())


if __name__ == "__main__":
    main()
