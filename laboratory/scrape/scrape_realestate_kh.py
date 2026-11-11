"""Polite, resumable scraper for realestate.com.kh house-for-sale listings (Phnom Penh).

Usage:
    python scripts/scrape_realestate_kh.py                 # everything, resumes automatically
    python scripts/scrape_realestate_kh.py --pages 5 --limit 20   # quick test
    python scripts/scrape_realestate_kh.py --category Apartment   # -> data/realestate_kh_apartment.csv

Behaviour:
- robots.txt allows /buy/ and listing pages (it disallows /api/, /v1/, /accounts/ ...,
  which this script never touches). Re-check it before big runs.
- Identifies itself honestly, waits --delay seconds (plus jitter) between requests.
- On HTTP 429 it waits (Retry-After if given, else 2/4/8 minutes) and retries,
  then stops. On 403 it stops immediately. It never tries to evade a block.
- Appends each row to the CSV as it goes and skips URLs already saved, so it can be
  stopped and restarted at any time.
- Price per m2 and area medians are NOT saved (they derive from price: leakage).
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.realestate.com.kh"
LIST_URL = BASE + "/buy/phnom-penh/"
HEADERS = {"User-Agent": "property-valuation-student-project (academic, low rate)"}
OUT = Path(__file__).resolve().parents[1] / "data" / "realestate_kh.csv"

COLUMNS = [
    "url", "listing_id", "title", "listing_type", "listed_raw", "updated_raw",
    "province", "district", "commune", "latitude", "longitude",
    "property_type", "borey_name", "title_deed", "facing", "floor_level",
    "price_usd", "bedrooms", "bathrooms", "car_spaces",
    "land_area_m2", "floor_area_m2",
    "land_dim_raw", "house_dim_raw", "land_width_m", "land_length_m",
    "house_width_m", "house_length_m",
    "property_features", "amenities", "description", "notes",
]

LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
NEXT_RE = re.compile(r'__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)
NUM = r"(\d+(?:[.,]\d+)?)"
DIM = rf"{NUM}\s*m?\s*[x×*]\s*{NUM}\s*m?"
LAND_DIM = re.compile(rf"land\s*(?:size|area)?\s*[:\-]?\s*{DIM}", re.I)
HOUSE_DIM = re.compile(rf"(?:house|building|floor)\s*(?:size|area)?\s*[:\-]?\s*{DIM}", re.I)


class Blocked(Exception):
    pass


def get(session: requests.Session, url: str, params=None) -> str:
    waits = [120, 240, 480]
    for attempt in range(len(waits) + 1):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < len(waits):
                print(f"  network error ({type(e).__name__}); waiting {waits[attempt]}s (retry {attempt + 1}/{len(waits)})", flush=True)
                time.sleep(waits[attempt])
                continue
            raise Blocked(f"network unavailable: {e!r}")
        if r.status_code == 429 and attempt < len(waits):
            wait = int(r.headers.get("Retry-After", 0) or 0) or waits[attempt]
            print(f"  429 rate limited; waiting {wait}s (retry {attempt + 1}/{len(waits)})", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (403, 429):
            raise Blocked(f"{r.status_code} from {r.url}; stopping instead of working around it")
        r.raise_for_status()
        return r.text
    raise Blocked("unreachable")


def list_urls(html: str) -> list[str]:
    for block in LD_RE.findall(html):
        if '"ItemList"' in block:
            return [i["item"]["url"] for i in json.loads(block)["itemListElement"]]
    return []


def badge(badges: list[dict], label_start: str):
    for b in badges:
        if b.get("label", "").lower().startswith(label_start.lower()):
            return b.get("displayValue") or ""
    return ""


def feature_value(features: list[dict], key: str) -> str:
    for group in features:
        for item in group.get("items", []):
            label = item.get("label", "")
            if label.lower().startswith(key.lower() + ":"):
                return label.split(":", 1)[1].strip()
    return ""


def feature_list(features: list[dict], headline: str) -> str:
    for group in features:
        if group.get("headline", "").lower() == headline.lower():
            return "; ".join(i.get("label", "") for i in group.get("items", []))
    return ""


def dims(regex: re.Pattern, text: str):
    m = regex.search(text)
    if not m:
        return "", "", ""
    w, l = (float(g.replace(",", ".")) for g in m.groups())
    return m.group(0).split(":")[-1].strip(), w, l


def digits(text) -> str:
    return re.sub(r"[^\d.]", "", text or "")


def parse_detail(url: str, html: str) -> dict:
    nd = json.loads(NEXT_RE.search(html).group(1))
    d = nd["props"]["pageProps"]["cacheData"]["listing"]["data"]
    header, badges = d["header"], d.get("badges", [])
    features = d.get("features", [])
    desc = (d.get("mainContent", {}).get("description") or "").replace("\r", "")
    aside = d.get("aside", {}).get("listing", {})
    parts = [p.strip() for p in (aside.get("address") or "").split(",")]
    # "Sangkat, Khan, Province"; shorter addresses drop the left side
    province = parts[-1] if parts else ""
    district = parts[-2] if len(parts) >= 2 else ""
    commune = parts[-3] if len(parts) >= 3 else ""

    lat = lng = ""
    for block in LD_RE.findall(html):
        if '"GeoCoordinates"' in block:
            geo = json.loads(block).get("geo", {})
            lat, lng = geo.get("latitude", ""), geo.get("longitude", "")

    land_raw, lw, ll = dims(LAND_DIM, desc)
    house_raw, hw, hl = dims(HOUSE_DIM, desc)
    notes = []
    if header.get("listingType") != "sale":
        notes.append(f"listingType={header.get('listingType')}")
    if aside.get("displayAsProject"):
        notes.append("project page, not a single unit")
    if header.get("status") != "current":
        notes.append(f"status={header.get('status')}")

    return {
        "url": url,
        "listing_id": re.search(r"-(\d+)/?$", url).group(1) if re.search(r"-(\d+)/?$", url) else "",
        "title": header.get("headline", ""),
        "listing_type": header.get("listingType", ""),
        "listed_raw": feature_value(features, "Listed"),
        "updated_raw": feature_value(features, "Updated"),
        "province": province, "district": district, "commune": commune,
        "latitude": lat, "longitude": lng,
        "property_type": (aside.get("categories") or [""])[0],
        "borey_name": (aside.get("project") or {}).get("projectName", ""),
        "title_deed": feature_value(features, "Title"),
        "facing": badge(badges, "Facing"),
        "floor_level": badge(badges, "Floor Level"),
        "price_usd": digits(header.get("salePrice")),
        "bedrooms": badge(badges, "Bedroom"),
        "bathrooms": badge(badges, "Bathroom"),
        "car_spaces": badge(badges, "Car") or badge(badges, "Garage"),
        "land_area_m2": badge(badges, "Land Area"),
        "floor_area_m2": badge(badges, "Floor Area"),
        "land_dim_raw": land_raw, "house_dim_raw": house_raw,
        "land_width_m": lw, "land_length_m": ll,
        "house_width_m": hw, "house_length_m": hl,
        "property_features": feature_list(features, "Property Features"),
        "amenities": feature_list(features, "Amenities"),
        "description": desc.replace("\n", " | ")[:2000],
        "notes": "; ".join(notes),
    }


def existing_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {r["url"] for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-page", type=int, default=0,
                    help="first list page (0 = resume from the page saved in the .page state file)")
    ap.add_argument("--pages", type=int, default=0, help="max list pages (0 = until empty)")
    ap.add_argument("--limit", type=int, default=0, help="max new detail pages (0 = all)")
    ap.add_argument("--delay", type=float, default=5.0, help="base seconds between requests")
    ap.add_argument("--category", default="House", help="realestate.com.kh category, e.g. House, Apartment")
    ap.add_argument("--out", type=Path, default=None, help="default: realestate_kh.csv for House, realestate_kh_<category>.csv otherwise")
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.category == "House" else f"_{args.category.lower()}"
        args.out = OUT.with_name(f"realestate_kh{suffix}.csv")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_urls(args.out)
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    print(f"{len(seen)} listings already saved", flush=True)

    def pause():
        time.sleep(args.delay + random.uniform(0, args.delay * 0.4))

    session = requests.Session()
    added = 0
    with args.out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        try:
            state = args.out.with_suffix(".page")
            if args.start_page:
                page = args.start_page
            elif state.exists():
                page = int(state.read_text().strip() or 1)
            else:
                page = 1
            args.start_page = page
            print(f"starting at list page {page}", flush=True)
            while not args.pages or page < args.start_page + args.pages:
                try:
                    html = get(session, LIST_URL, {"categories": args.category, "page": page})
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        print(f"page {page}: 404, past the last page; finished", flush=True)
                        break
                    raise
                urls = list_urls(html)
                todo = [u for u in urls if u not in seen]
                print(f"page {page}: {len(urls)} listings, {len(todo)} new", flush=True)
                if not urls:
                    break
                for url in todo:
                    if args.limit and added >= args.limit:
                        raise StopIteration
                    pause()
                    try:
                        w.writerow(parse_detail(url, get(session, url)))
                        f.flush()
                        seen.add(url)
                        added += 1
                    except Blocked:
                        raise
                    except Exception as e:  # keep going on a single bad page
                        print(f"  skip {url}: {e!r}", flush=True)
                page += 1
                state.write_text(str(page))  # next run resumes here
                pause()
        except Blocked as e:
            print(e, file=sys.stderr)
        except StopIteration:
            pass
    print(f"done: added {added} rows, {len(seen)} total in {args.out}", flush=True)


if __name__ == "__main__":
    main()
