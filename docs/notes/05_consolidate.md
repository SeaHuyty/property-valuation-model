# Preprocess scraped data

- **Date:** 2026-11-22 20:45
- **Commit:** `d447b5a`
- **Author:** SeaHuyty

## What
- `consolidate_listings.py`: union of all site columns into one schema, repair a corrupted CSV row, dedupe by URL → `listings_consolidated.csv` (~13,880 rows).
- `extract_title_features.py`: bedrooms, bathrooms, floor, title deed from titles (English + Khmer); fix mislabeled types; catch rentals posted as sales.
- `extract_description_features.py`: land/house dimensions, rooms, title deed, floor area from descriptions (93–100% agreement with structured values).
- Raw scrapes moved to `data/rawr/`; first `data/clean/data.csv`.

## Why / result
Three sites, three formats → one table the model can use.

## Files
- `laboratory/scripts/consolidate_listings.py`
- `laboratory/scripts/extract_*_features.py`
- `laboratory/data/consolidated/`, `data/clean/data.csv`
