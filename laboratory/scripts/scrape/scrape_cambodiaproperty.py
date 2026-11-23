"""Polite, resumable scraper for cambodiaproperty.asia house-for-sale listings (Phnom Penh).

Usage:
    python scripts/scrape_cambodiaproperty.py
    python scripts/scrape_cambodiaproperty.py --limit 5     # quick test

Behaviour:
- robots.txt (checked 2026-11-09) disallows only named crawlers plus /private/, /dashboard/,
  /wp-admin/ etc. for everyone else; search and listing pages are allowed and there is no
  AI-training signal. Re-check it before big runs.
- Honest User-Agent, --delay seconds (plus jitter) between requests, stops on 403/429 after
  backing off. Never tries to evade a block.
- Appends each row and skips URLs already saved (resumable).
- Price per m2 is NOT saved (derives from price: leakage).
- Listing data comes from the page's ld+json `RealEstateListing` block plus the description.
  The site's "Area" is ambiguous (usually land); we keep it as `area_m2` and also parse
  "W x L" dimensions from the description.
"""
import argparse
import csv
import html
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.cambodiaproperty.asia"
SEARCH = BASE + "/en/property-search/"
HEADERS = {"User-Agent": "property-valuation-student-project (academic, low rate)"}
OUT = Path(__file__).resolve().parents[1] / "data" / "cambodiaproperty.csv"

COLUMNS = [
    "url", "listing_id", "title", "listed_raw", "province", "district", "commune",
    "latitude", "longitude", "property_type", "price_usd", "bedrooms", "bathrooms",
    "area_m2", "dim_raw", "width_m", "length_m", "features", "description", "notes",
]

LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
NUM = r"(\d+(?:[.,]\d+)?)"
DIM_RE = re.compile(rf"{NUM}\s*m?\s*[x×*]\s*{NUM}\s*m?", re.I)
TYPES = [  # order matters: most specific first
    ("twin villa", "Twin Villa"), ("shophouse", "Shophouse"), ("shop house", "Shophouse"),
    ("townhouse", "Townhouse"), ("link house", "Link House"), ("flat", "Flat"),
    ("villa", "Villa"), ("condo", "Condo"), ("apartment", "Apartment"),
    ("land", "Land"), ("house", "House"),
]


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


def list_urls(page_html: str) -> list[str]:
    seen, out = set(), []
    for u in re.findall(r"https://www\.cambodiaproperty\.asia/en/property/[^\"'\s<>]+", page_html):
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def guess_type(title: str) -> str:
    low = title.lower()
    for key, name in TYPES:
        if key in low:
            return name
    return ""


def prop(ld: dict, name: str) -> str:
    for p in ld.get("additionalProperty", []):
        if p.get("name", "").lower() == name.lower():
            return str(p.get("value", ""))
    return ""


def parse_detail(url: str, page_html: str) -> dict:
    ld = {}
    published = ""
    for block in LD_RE.findall(page_html):
        if '"RealEstateListing"' in block:
            ld = json.loads(block)
            break
    m = re.search(r'"datePublished":"([^"]+)"', page_html)
    if m:
        published = m.group(1)[:10]

    addr = ld.get("address", {}).get("streetAddress", "")
    parts = [p.strip() for p in addr.split(",")]
    commune = district = ""
    for p in parts:
        if p.lower().startswith("sangkat") or p.lower().startswith("commune"):
            commune = re.sub(r"^(sangkat|commune)\s+", "", p, flags=re.I)
        elif p.lower().startswith("khan") or p.lower().startswith("district"):
            district = re.sub(r"^(khan|district)\s+", "", p, flags=re.I)
    geo = ld.get("geo", {})

    # description + feature list as plain text
    body = re.sub(r"<(script|style).*?</\1>", "", page_html, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", body))
    text = re.sub(r"\n\s*\n+", "\n", text)
    d0 = text.find("Description")
    d1 = text.find("Features", d0)
    desc = text[d0 + len("Description"):d1].strip() if d0 >= 0 and d1 > d0 else ld.get("description", "")
    f1 = text.find("Explore similar properties", d1)
    features = "; ".join(l.strip() for l in text[d1 + len("Features"):f1].splitlines() if l.strip()) if d1 >= 0 and f1 > d1 else ""

    dim = DIM_RE.search(desc)
    w = l = ""
    if dim:
        w, l = (float(g.replace(",", ".")) for g in dim.groups())
    title = ld.get("name", "")
    notes = []
    if "/availability" in json.dumps(ld.get("offers", {})) and "Sale" not in ld.get("offers", {}).get("availability", ""):
        notes.append("not a sale offer")
    if not re.search(r"-\d+/?$", url):
        notes.append("project/development page, not a single unit")
    if not prop(ld, "Bedrooms"):
        notes.append("no bedrooms (land/project page?)")

    return {
        "url": url,
        "listing_id": (re.search(r"-(\d+)/?$", url) or [None, ""])[1],
        "title": title,
        "listed_raw": published,
        "province": "Phnom Penh",  # search is filtered to Phnom Penh; addressLocality is unreliable
        "district": district, "commune": commune,
        "latitude": geo.get("latitude", ""), "longitude": geo.get("longitude", ""),
        "property_type": guess_type(title),
        "price_usd": re.sub(r"[^\d.]", "", str(ld.get("offers", {}).get("price", ""))),
        "bedrooms": prop(ld, "Bedrooms"), "bathrooms": prop(ld, "Bathrooms"),
        "area_m2": prop(ld, "Area Size"),
        "dim_raw": dim.group(0) if dim else "", "width_m": w, "length_m": l,
        "features": features,
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
    ap.add_argument("--limit", type=int, default=0, help="max new detail pages (0 = all)")
    ap.add_argument("--delay", type=float, default=5.0, help="base seconds between requests")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_urls(args.out)
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    print(f"{len(seen)} listings already saved", flush=True)

    def pause():
        time.sleep(args.delay + random.uniform(0, args.delay * 0.4))

    params = {"status": "sale", "location[0]": "phnom-penh-real-estate", "type[0]": "house"}
    session = requests.Session()
    added = 0
    with args.out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        try:
            page, max_page = 1, 1
            while page <= max_page:
                url = SEARCH if page == 1 else f"{SEARCH}page/{page}/"
                page_html = get(session, url, params)
                m = re.search(r'data-max="(\d+)"', page_html)
                max_page = int(m.group(1)) if m else 1
                urls = list_urls(page_html)
                todo = [u for u in urls if u not in seen]
                print(f"page {page}/{max_page}: {len(urls)} listings, {len(todo)} new", flush=True)
                for u in todo:
                    if args.limit and added >= args.limit:
                        raise StopIteration
                    pause()
                    try:
                        w.writerow(parse_detail(u, get(session, u)))
                        f.flush()
                        seen.add(u)
                        added += 1
                    except Blocked:
                        raise
                    except Exception as e:
                        print(f"  skip {u}: {e!r}", flush=True)
                page += 1
                pause()
        except Blocked as e:
            print(e, file=sys.stderr)
        except StopIteration:
            pass
    print(f"done: added {added} rows, {len(seen)} total in {args.out}", flush=True)


if __name__ == "__main__":
    main()
