"""Profile the hand-collected sample to help decide model features.

Usage: python scripts/profile_sample.py [path/to/sample_listings.csv]

Size columns (land_size_raw, building_size_raw) may hold a total area
("120", "120 m2") or width x length ("4.2m x 16m", "4.2x16"). Both are
parsed into square meters.
"""
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "sample_listings.csv"
NUM = r"(\d+(?:[.,]\d+)?)"
DIM_RE = re.compile(rf"{NUM}\s*(?:m|meters?)?\s*[x×*]\s*{NUM}", re.I)
AREA_RE = re.compile(NUM)


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def parse_area(value) -> float | None:
    """Return area in m2 from 'W x L' or a single total-area value."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    m = DIM_RE.search(text)
    if m:
        return _f(m.group(1)) * _f(m.group(2))
    m = AREA_RE.search(text)
    return _f(m.group(1)) if m else None


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = pd.read_csv(path)
    if df.empty:
        sys.exit(f"{path} has no rows yet.")

    df["area_m2"] = df["area_raw"].map(parse_area)
    df["land_m2"] = df["land_size_raw"].map(parse_area)
    df["building_m2"] = df["building_size_raw"].map(parse_area)

    print(f"Rows: {len(df)}\n")
    print("Missing rate per column (%):")
    print((df.isna().mean() * 100).round(1).sort_values(ascending=False).to_string(), "\n")
    print("Rows per property type:")
    print(df["property_type"].value_counts(dropna=False).to_string(), "\n")
    print("Rows per province:")
    print(df["province"].value_counts(dropna=False).to_string(), "\n")
    print("Rows per source:")
    print(df["source"].value_counts(dropna=False).to_string(), "\n")
    print("Area type (is area land or building?):")
    print(df["area_type"].value_counts(dropna=False).to_string(), "\n")
    print("Numeric summary:")
    print(df[["price_usd", "area_m2", "land_m2", "building_m2", "bedrooms", "bathrooms", "floors"]]
          .describe().round(1).to_string())

    parts = []
    for raw, parsed in [("area_raw", "area_m2"), ("land_size_raw", "land_m2"),
                        ("building_size_raw", "building_m2")]:
        parts.append(df[df[raw].notna() & df[parsed].isna()][[raw]])
    unparsed = pd.concat(parts)
    if not unparsed.empty:
        print("\nSize strings that could not be parsed:")
        print(unparsed.to_string())


if __name__ == "__main__":
    main()
