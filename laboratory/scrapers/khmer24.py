"""Khmer24 property scraper for the property-valuation-model project.

Three stages, each resumable (re-running skips work already saved):

  1. feed     Page through Khmer24's listing feed API (30 listings/request).
              Gives: id, title, price, province/district/commune, bedrooms,
              bathrooms, size.                      -> data/raw/khmer24_feed.jsonl
  2. details  Open each listing page and read the JSON embedded in it
              (__NUXT_DATA__). Gives: property type (Flat/House/Shophouse/
              Villa), posted date, description text.  -> data/raw/khmer24_details.jsonl
  3. export   Merge both, parse sizes/floors/borey from the description and
              write the same columns as data/sample_listings.csv.
                                                     -> data/khmer24_listings.csv

Usage (run from the laboratory/ folder):
  python scrapers/khmer24.py feed                # houses + condos (default)
  python scrapers/khmer24.py details --limit 500        # try a small batch first
  python scrapers/khmer24.py export

Be polite: keep the default delays, run during off-peak hours, and stop if you
start getting 403/429 responses instead of trying to get around them.
Seller names/phone fields are never written to disk. Descriptions in data/raw/
can still contain phone numbers sellers typed in, so data/raw/ is git-ignored;
the exported CSV has phone numbers stripped from titles.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

LAB_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = LAB_DIR / "data" / "raw"
FEED_FILE = RAW_DIR / "khmer24_feed.jsonl"
DETAIL_FILE = RAW_DIR / "khmer24_details.jsonl"
OUT_CSV = LAB_DIR / "data" / "khmer24_listings.csv"

FEED_URL = "https://api-posts.khmer24.com/feed"
DETAIL_URL = "https://www.khmer24.com/en/post-adid-{id}"
FEED_FIELDS = "location,category,link,object_highlight_specs,condition,renew_date"
DEFAULT_CATEGORIES = ["house-for-sale", "condo-for-sale"]  # add "land-for-sale" to include land

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Fields dropped from detail records so no personal data is stored.
PRIVATE_KEYS = {"user", "store", "phone", "phone_white_operator", "photos",
                "thumbnails", "thumbnail", "photo"}


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def polite_get(session: requests.Session, url: str, delay: float, **kw) -> requests.Response | None:
    """GET with a randomized delay and a few retries on transient errors.

    On 403/429 it waits longer and finally gives up (returns None) rather than
    trying to get around the block.
    """
    for attempt in range(4):
        time.sleep(delay + random.uniform(0, delay * 0.5))
        try:
            r = session.get(url, timeout=30, **kw)
        except requests.RequestException as e:
            print(f"  network error ({e.__class__.__name__}), retry {attempt + 1}/4")
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429, 503):
            wait = 60 * (attempt + 1)
            print(f"  HTTP {r.status_code} - the site is asking us to slow down; waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return None
        print(f"  HTTP {r.status_code} for {url}")
        return None
    print("  Giving up on this URL. If this keeps happening, stop and increase --delay.")
    return None


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Stage 1: feed
# --------------------------------------------------------------------------- #
def scrape_feed(categories: list[str], delay: float, max_pages: int | None) -> None:
    session = make_session()
    seen = {r["id"] for r in read_jsonl(FEED_FILE)}
    print(f"{len(seen)} listings already saved in {FEED_FILE.name}")

    for cat in categories:
        offset, page, new_total = 0, 0, 0
        while True:
            params = {"category": cat, "offset": offset, "lang": "en",
                      "meta": "true", "fields": FEED_FIELDS}
            r = polite_get(session, FEED_URL, delay, params=params)
            if r is None:
                break
            payload = r.json()
            posts = [x["data"] for x in payload.get("data", []) if x.get("type") == "post"]
            if not posts:
                break

            rows = []
            for p in posts:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                p.pop("user", None)  # seller info: not needed, not stored
                for k in ("photos", "thumbnails", "thumbnail", "store"):
                    p.pop(k, None)
                p["_category"] = cat
                rows.append(p)
            append_jsonl(FEED_FILE, rows)
            new_total += len(rows)

            page += 1
            offset += int(payload.get("limit") or 30)
            total = payload.get("total", "?")
            print(f"[{cat}] page {page}  offset {offset}/{total}  +{len(rows)} new")
            if max_pages and page >= max_pages:
                break
        print(f"[{cat}] done: {new_total} new listings\n")


# --------------------------------------------------------------------------- #
# Stage 2: details
# --------------------------------------------------------------------------- #
NUXT_RE = re.compile(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', re.S)
_WRAPPERS = {"ShallowReactive", "Reactive", "Ref", "ShallowRef", "EmptyRef", "EmptyShallowRef"}


def decode_nuxt_data(raw: str):
    """Decode Nuxt 3's __NUXT_DATA__ payload (a flat array where values point
    to other indexes) back into normal nested dicts/lists."""
    arr = json.loads(raw)
    memo: dict[int, object] = {}

    def h(i):
        if not isinstance(i, int) or isinstance(i, bool):
            return i
        if i < 0:
            return None
        if i in memo:
            return memo[i]
        v = arr[i]
        if isinstance(v, list):
            if v and isinstance(v[0], str) and v[0] in _WRAPPERS:
                out = h(v[1])
                memo[i] = out
                return out
            out_l: list = []
            memo[i] = out_l
            out_l.extend(h(x) for x in v)
            return out_l
        if isinstance(v, dict):
            out_d: dict = {}
            memo[i] = out_d
            for k, x in v.items():
                out_d[k] = h(x)
            return out_d
        memo[i] = v
        return v

    return h(0)


def extract_post(html: str, post_id: str) -> dict | None:
    m = NUXT_RE.search(html)
    if not m:
        return None
    root = decode_nuxt_data(m.group(1))
    for value in (root.get("data") or {}).values():
        post = value.get("data") if isinstance(value, dict) else None
        if isinstance(post, dict) and str(post.get("id")) == str(post_id):
            return {k: v for k, v in post.items() if k not in PRIVATE_KEYS}
    return None


def scrape_details(delay: float, limit: int | None, categories: list[str] | None) -> None:
    session = make_session()
    feed = read_jsonl(FEED_FILE)
    if not feed:
        sys.exit("Run the 'feed' stage first.")
    done = {r["id"] for r in read_jsonl(DETAIL_FILE)}
    todo = [r["id"] for r in feed
            if r["id"] not in done and (not categories or r.get("_category") in categories)]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} already done, {len(todo)} to fetch "
          f"(~{len(todo) * delay * 1.25 / 3600:.1f} h at --delay {delay})")

    failures = 0
    for n, pid in enumerate(todo, 1):
        r = polite_get(session, DETAIL_URL.format(id=pid), delay)
        post = extract_post(r.text, pid) if r is not None else None
        if post is None:
            failures += 1
            append_jsonl(DETAIL_FILE, [{"id": pid, "_missing": True}])
            if failures >= 20 and failures / n > 0.5:
                sys.exit("More than half of the detail pages are failing. Stopping - "
                         "wait a while and re-run with a larger --delay.")
        else:
            append_jsonl(DETAIL_FILE, [post])
        if n % 25 == 0:
            print(f"  {n}/{len(todo)} fetched ({failures} failed)")
    print("details done")


# --------------------------------------------------------------------------- #
# Stage 3: export to the project's CSV schema
# --------------------------------------------------------------------------- #
KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
NUM = r"\d+(?:[.,]\d+)?"
METER = r"(?:\s*(?:m|M|ម៉ែត្រ|ម|meters?|metres?))?"
DIM = rf"{NUM}{METER}\s*[xX×*]\s*{NUM}{METER}"
SQ_UNIT = r"(?:m2|m²|sqm|sq\.?\s*m|ម៉ែត្រការ៉េ|ម៉ែត្រការេ|ម៉ែត្រក្រឡា|ម២|㎡)"
AREA = rf"{NUM}\s*{SQ_UNIT}"
# "ទទឹង 5m បណ្តោយ 13m" / "width 5m length 13m"
WIDTH_LENGTH = (rf"(?:ទទឹង|width)\s*[:៖]?\s*({NUM}){METER}[^\d\n]{{0,8}}"
                rf"(?:បណ្តោយ|បណ្ដោយ|length)\s*[:៖]?\s*({NUM}){METER}")
SIZE_VALUE = rf"({DIM}|{AREA}|{WIDTH_LENGTH})"

LAND_LABELS = (r"(?:ទំហំដី|ដីទំហំ|ផ្ទៃដី|សាច់ដី|ដី|land\s*size|land\s*area|"
               r"size\s*of\s*land|land|plot)")
BUILDING_LABELS = (r"(?:ទំហំផ្ទះ|ផ្ទះទំហំ|ទំហំអគារ|ផ្ទៃក្រឡាផ្ទះ|ផ្ទៃប្រើប្រាស់|អគារ|"
                   r"(?:house|villa|flat|shophouse|building|home)\s*size|"
                   r"built[-\s]*up\s*area|floor\s*area|living\s*area|house)")
LABEL_GAP = r"[^\d\n]{0,15}"  # e.g. "៖ ", ": ", " about "

LAND_RE = re.compile(LAND_LABELS + LABEL_GAP + SIZE_VALUE, re.I)
BUILDING_RE = re.compile(BUILDING_LABELS + LABEL_GAP + SIZE_VALUE, re.I)
WIDTH_LENGTH_RE = re.compile(WIDTH_LENGTH, re.I)
FLOORS_RE = re.compile(rf"({NUM})\s*(?:ជាន់|floors?|stor(?:e)?ys?|levels?)(?![a-z])", re.I)
G_PLUS_RE = re.compile(r"\bG\s*\+\s*(\d+)", re.I)
BOREY_RE = re.compile(r"(?:បុរី|borey)\s*([^\s,.។()\n|]+(?:[ \t]+[^\s,.។()\n|]+){0,3})", re.I)
# Text that follows a borey name but isn't part of it ("urgent sale", "house no.", ...)
BOREY_STOP_RE = re.compile(r"\s*(?:ត្រូវការ|លក់|ផ្ទះលេខ|ផ្លូវ|ទើប|គំរោង|តម្លៃ|ទីតាំង|—|-|–|"
                           r"\b(?:for\s+sale|urgent|price|near)\b|\d+\s*(?:storey|floor)).*", re.I)
NEGOTIABLE_RE = re.compile(r"negotiable|ចរចា|ចចារ", re.I)
# Cambodian phone numbers sellers type into titles: 012 345 678, 0961234567, +855 12 345 678
PHONE_RE = re.compile(r"(?<!\d)(?:(?:0\d{8,9}){2,}"  # several numbers glued together
                      r"|(?:\+?855[\s.-]?0?|0)\d{2}[\s.-]?\d{3}[\s.-]?\d{3,4})(?!\d)")


def norm(text: str | None) -> str:
    return (text or "").translate(KHMER_DIGITS).replace("​", " ")


def strip_phones(text: str) -> str:
    """Remove phone numbers so exported text holds no personal contact data."""
    text = PHONE_RE.sub("", norm(text))
    text = re.sub(r"(?:Tel|Phone|Call|ទូរស័ព្ទ|លេខទូរស័ព្ទ)\s*[:៖.]?\s*(?=[\s/,|]*$)", "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip(" -/,|:")


def clean_size(raw: str | None) -> str:
    """'4.2ម x 18.5ម' -> '4.2m x 18.5m', 'ទទឹង 5m បណ្តោយ 13m' -> '5m x 13m',
    so profile_sample.py's parse_area() reads it."""
    if not raw:
        return ""
    m = WIDTH_LENGTH_RE.search(raw)
    if m:
        return f"{m.group(1)}m x {m.group(2)}m"
    raw = re.sub(r"ម៉ែត្រការ៉េ|ម៉ែត្រការេ|ម៉ែត្រក្រឡា|ម២|㎡", "m2", raw)
    return re.sub(r"ម៉ែត្រ|ម", "m", raw).strip()


def parse_borey(*texts: str) -> str:
    name = first_match(BOREY_RE, *texts) or ""
    name = BOREY_STOP_RE.sub("", name)
    return re.sub(r"\s+", " ", name).strip(" :៖")


def first_match(regex: re.Pattern, *texts: str) -> str | None:
    for t in texts:
        m = regex.search(t)
        if m:
            return m.group(1).strip()
    return None


def parse_floors(text: str) -> int | None:
    m = G_PLUS_RE.search(text)
    if m:
        return int(m.group(1)) + 1
    m = FLOORS_RE.search(text)
    if m:
        n = float(m.group(1).replace(",", "."))
        if 1 <= n <= 40:
            return int(n)
    return None


def split_location(loc: dict) -> tuple[str, str, str]:
    """long_location is 'Commune, District, Province' (shorter when missing)."""
    parts = [p.strip() for p in (loc.get("long_location") or "").split(",") if p.strip()]
    province = loc.get("en_name") or (parts[-1] if parts else "")
    district = parts[-2] if len(parts) >= 2 else ""
    commune = parts[-3] if len(parts) >= 3 else ""
    return province, district, commune


def spec(post: dict, key: str) -> str | None:
    for container in ("object_specs", "object_highlight_specs"):
        item = (post.get(container) or {}).get(key)
        if isinstance(item, dict) and item.get("value") not in (None, ""):
            return str(item["value"])
    return None


def to_int(value: str | None) -> int | None:
    if value is None:
        return None
    m = re.search(r"\d+", value)
    return int(m.group()) if m else None


def build_row(feed: dict, detail: dict | None) -> dict | None:
    post = {**feed, **(detail or {})}
    if (post.get("condition") or {}).get("value", "sale") != "sale":
        return None  # skip rentals that slip into sale categories

    cat = feed.get("_category", "")
    title = post.get("title") or ""
    desc = norm(post.get("description"))
    title_n = norm(title)
    province, district, commune = split_location(post.get("location") or {})

    if cat == "land-for-sale":
        property_type, area_type = "Land", "land"
    elif cat == "condo-for-sale":
        property_type, area_type = spec(post, "type") or "Condo", "building"
    else:
        property_type, area_type = spec(post, "type") or "House (unspecified)", "unspecified"

    land_raw = first_match(LAND_RE, desc, title_n)
    building_raw = first_match(BUILDING_RE, desc, title_n)

    notes = []
    if detail is None or detail.get("_missing"):
        notes.append("no_detail")
    if NEGOTIABLE_RE.search(desc + " " + title_n):
        notes.append("negotiable")

    try:
        price = float(post.get("price") or 0) or None
    except ValueError:
        price = None

    return {
        "source": "khmer24",
        "url": post.get("link") or DETAIL_URL.format(id=post["id"]),
        "title": strip_phones(title.replace("\n", " ")),
        "listed_raw": post.get("posted_date") or post.get("renew_date") or "",
        "province": province,
        "district": district,
        "commune": commune,
        "property_type": property_type,
        "borey_name": strip_phones(parse_borey(desc, title_n)),
        "price_usd": price,
        "area_raw": spec(post, "size") or "",
        "area_type": area_type,
        "land_size_raw": clean_size(land_raw),
        "building_size_raw": clean_size(building_raw),
        "bedrooms": to_int(spec(post, "bedroom")),
        "bathrooms": to_int(spec(post, "bathroom")),
        "floors": parse_floors(desc) or parse_floors(title_n),
        "notes": ";".join(notes),
    }


def export_csv(categories: list[str]) -> None:
    import pandas as pd

    feed = [f for f in read_jsonl(FEED_FILE) if f.get("_category") in categories]
    if not feed:
        sys.exit("Nothing to export - run the 'feed' stage first.")
    details = {r["id"]: r for r in read_jsonl(DETAIL_FILE)}
    rows = [row for f in feed if (row := build_row(f, details.get(f["id"]))) is not None]

    columns = pd.read_csv(LAB_DIR / "data" / "sample_listings.csv", nrows=0).columns.tolist()
    df = pd.DataFrame(rows).reindex(columns=columns)
    df = df.drop_duplicates(subset="url")
    for col in ("bedrooms", "bathrooms", "floors"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows Khmer

    have_detail = sum(1 for f in feed if f["id"] in details and not details[f["id"]].get("_missing"))
    print(f"Wrote {len(df)} rows to {OUT_CSV}")
    print(f"  with detail page: {have_detail}")
    for col in ("land_size_raw", "building_size_raw", "floors", "borey_name"):
        filled = int((df[col].notna() & (df[col].astype(str).str.strip() != "")).sum())
        print(f"  {col:18s} filled in {filled} rows ({filled / max(len(df), 1):.0%})")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    f = sub.add_parser("feed", help="stage 1: listing feed API")
    f.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    f.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    f.add_argument("--max-pages", type=int, help="stop after N pages per category (for testing)")

    d = sub.add_parser("details", help="stage 2: individual listing pages")
    d.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    d.add_argument("--limit", type=int, help="only fetch this many (run again to continue)")
    d.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)

    e = sub.add_parser("export", help="stage 3: write data/khmer24_listings.csv")
    e.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)

    a = ap.parse_args()
    if a.stage == "feed":
        scrape_feed(a.categories, a.delay, a.max_pages)
    elif a.stage == "details":
        scrape_details(a.delay, a.limit, a.categories)
    else:
        export_csv(a.categories)


if __name__ == "__main__":
    main()
