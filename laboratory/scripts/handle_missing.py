"""Handle missing values in data/clean/data.csv.

Usage: python scripts/handle_missing.py

Reads  data/clean/data_unimputed.csv if present, else data/clean/data.csv
       (and saves that as data_unimputed.csv first, so the step is re-runnable).
Writes data/clean/data.csv

  floors                dropped (82% missing for houses, not applicable to condos)
  borey_name            missing -> "No Borey" (listing is not in a borey)
  commune               missing -> "Unknown"
  not applicable        land_width_m / land_length_m for condos -> 0
                        floor_level for non-condos -> 0;  bedrooms/bathrooms for Land -> 0
  genuinely missing     area_m2, bedrooms, bathrooms, land_width_m, land_length_m, floor_level
                        -> hierarchical group median, plus a <col>_missing 0/1 flag

Group medians are computed on the whole dataset. For a strict evaluation, fit the
same medians on the training split only (see docs/dataset.md).
"""
from pathlib import Path

import pandas as pd

CLEAN = Path(__file__).resolve().parents[1] / "data" / "clean"
SRC, RAW = CLEAN / "data.csv", CLEAN / "data_unimputed.csv"


def impute(d, col, levels, mask, integer=False):
    """Fill d[col] where mask & missing with the median of the first group level
    (most specific first) that has a value."""
    todo = mask & d[col].isna()
    d[f"{col}_missing"] = todo.astype(int)
    for keys in levels:
        fill = todo & d[col].isna()
        if not keys:
            d.loc[fill, col] = d.loc[mask, col].median()
            continue
        med = d[mask].groupby(keys)[col].median().rename("_m")
        looked_up = d[keys].join(med, on=keys)["_m"]
        d.loc[fill, col] = looked_up[fill]
    if integer:
        d.loc[todo, col] = d.loc[todo, col].round()
    return int(todo.sum())


def main() -> None:
    if not RAW.exists():
        pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False).to_csv(RAW, index=False, encoding="utf-8-sig")
    d = pd.read_csv(RAW, encoding="utf-8-sig", low_memory=False)

    d = d.drop(columns=["floors"], errors="ignore")
    d["borey_name"] = d["borey_name"].fillna("No Borey")
    d["commune"] = d["commune"].fillna("Unknown")

    condo = d["property_type"].eq("Condo")
    land = d["property_type"].eq("Land")
    d.loc[condo, ["land_width_m", "land_length_m"]] = 0
    d.loc[~condo, "floor_level"] = 0
    d.loc[land, ["bedrooms", "bathrooms"]] = d.loc[land, ["bedrooms", "bathrooms"]].fillna(0)

    # helper bins for grouping
    d["_bed"] = d["bedrooms"].clip(upper=6)
    log = {}
    every = pd.Series(True, index=d.index)
    log["area_m2"] = impute(d, "area_m2",
                            [["property_type", "district", "_bed"], ["property_type", "district"],
                             ["property_type", "_bed"], ["property_type"]], every)
    d["_area_q"] = d.groupby("property_type")["area_m2"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False) if s.notna().sum() >= 4 else 0)
    log["bedrooms"] = impute(d, "bedrooms",
                             [["property_type", "district", "_area_q"], ["property_type", "_area_q"],
                              ["property_type"]], every, integer=True)
    d["_bed"] = d["bedrooms"].clip(upper=6)
    log["bathrooms"] = impute(d, "bathrooms",
                              [["property_type", "_bed"], ["property_type"]], every, integer=True)
    for col in ("land_width_m", "land_length_m"):
        log[col] = impute(d, col, [["property_type", "district", "_area_q"], ["property_type", "_area_q"],
                                   ["property_type"]], ~condo)
    log["floor_level"] = impute(d, "floor_level", [["district"], []], condo, integer=True)

    d = d.drop(columns=["_bed", "_area_q"])
    # one flag is enough for the two land dims (always missing together)
    d["land_dims_missing"] = d.pop("land_width_m_missing") | d.pop("land_length_m_missing")
    d.to_csv(SRC, index=False, encoding="utf-8-sig")

    for k, v in log.items():
        print(f"imputed {v:>5}  {k}")
    print(f"remaining missing cells: {int(d.isna().sum().sum())}")
    print(f"wrote {len(d)} rows x {d.shape[1]} cols -> {SRC}")


if __name__ == "__main__":
    main()
