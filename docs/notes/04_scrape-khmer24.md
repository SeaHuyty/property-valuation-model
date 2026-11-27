# Add Khmer24 scraper and dataset

- **Date:** 2026-11-12 12:46
- **Commit:** `e8475da`
- **Author:** Phay Someth

## What
- Resumable 3-stage scraper for khmer24.com: feed → detail pages → export, same columns as the sample listings.
- Export strips phone numbers from titles and borey names; `data/raw/` ignored (large, has seller phones).
- 7,645 house/condo listings (3,018 with detail pages at that point).

## Why / result
khmer24 is the biggest source (mostly condos, few details).

## Files
- `laboratory/scrapers/khmer24.py`
- `laboratory/data/khmer24_listings.csv`
- `.gitignore`, `requirements.txt`
