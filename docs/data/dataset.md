# Dataset

Training data for the price model: `laboratory/data/clean/data.csv`
(12,491 sale listings × 25 columns, no missing values). 95% Phnom Penh.

## Sources

| Site | Raw rows | Notes |
|---|---|---|
| khmer24.com | 7,645 | Mostly condos, few details, no descriptions |
| realestate.com.kh (house + apartment) | 6,197 | Rich structured fields + descriptions |
| cambodiaproperty.asia | 39 | Small; only source with coordinates (dropped) |

Raw scrapes: `laboratory/data/rawr/`. Merged, untouched union: `laboratory/data/consolidated/listings_consolidated.csv`.

## Pipeline

Run from `laboratory/`, in order:

1. `scripts/consolidate_listings.py` – union all site columns, repair one corrupted CSV row, dedupe by URL.
2. Manual drops: source/url/id columns, rent-only listings, date and facing columns.
3. `scripts/extract_title_features.py` – fill bedrooms, bathrooms, floor level, title deed from titles (English + Khmer); fix mislabeled types (realestate `Retreat` = Flat, `Unit` = Link House); detect rentals posted as sales.
4. `scripts/extract_description_features.py` – fill land/house dimensions, bedrooms, bathrooms, title deed, floor area from descriptions. Parsed values agree with existing structured values 93–100% (floor area 84%: sites mix net and gross).
5. `scripts/clean_dataset.py` – drop project pages, bad prices (< $5k, > $20M, impossible $/m²), duplicates; merge size columns; group property types; unify district/commune spellings; turn amenity lists into flags; null impossible values.
6. Manual: keep one area column, drop low-value flags.
7. `scripts/handle_missing.py` – handle missing values (below), then drop 56 rows whose imputed area gives an impossible price/m² (e.g. $18M "condos"). Saves the pre-imputation table as `data_unimputed.csv`.

## Columns

| Column | Meaning |
|---|---|
| `price_usd` | Asking price (target). Log-transform before training. |
| `property_type` | Condo, Villa, Flat, House, Shophouse, Twin Villa, Link House, Land |
| `province`, `district`, `commune` | Location; spellings unified (14 Phnom Penh districts) |
| `borey_name`, `in_borey` | Gated community / project name in English, one per developer or project (e.g. all "ប៉េងហួតបឹងស្នោរ …" and "Borey Peng Huoth: The Star …" variants → "Peng Huoth"); "No Borey" if none or when the scraped text was not a name (then `in_borey` = 0). Normalized by `scripts/normalize_borey.py` (rules in `engines/borey.py`) — 1,094 raw values → 181. 0/1 flag |
| `title_deed` | Hard / Strata / Soft / Long-Term Leasehold / Unknown |
| `area_m2` | **Condo: unit floor area. All other types: land (plot) area.** |
| `land_width_m`, `land_length_m` | Plot frontage × depth (house dims as fallback). 0 for condos. |
| `floor_level` | Condo's floor. 0 for non-condos. |
| `bedrooms`, `bathrooms` | Counts. 0 for land. |
| `has_gym`, `has_lift`, `has_parking`, `has_pool`, `has_sauna`, `on_main_road` | Amenity flags (0/1) |
| `*_missing` | 1 = value was imputed (`area_m2`, `bedrooms`, `bathrooms`, `floor_level`, `land_dims`) |

## Missing values

| Case | Handling |
|---|---|
| Not applicable (condo land dims, house floor level, land bedrooms) | 0 |
| `floors` (82% missing for houses) | Dropped |
| `borey_name` / `commune` empty | "No Borey" / "Unknown" |
| `area_m2` (20%), `bedrooms` (22%), `bathrooms` (37%), land dims, condo `floor_level` | Median of the most specific matching group (type + district + size/bedroom bucket, falling back to type), plus a `_missing` flag |

## Caveats

- **Imputation leakage:** medians are computed on the full dataset. For final evaluation, split first and fit medians on the training set only (re-impute from `data_unimputed.csv`).
- **Amenity flags are incomplete for khmer24 rows** (no description or amenity list), so 0 can mean "unknown". The source column was removed, so this can't be separated anymore.
- **Site price gap:** realestate.com.kh condos list ~28% higher per m² than khmer24 for the same areas.
- **Land:** only 16 rows, too few to learn from.
- **Asking prices**, not transaction prices.
