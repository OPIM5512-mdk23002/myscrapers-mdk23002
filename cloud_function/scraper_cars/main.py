# cloud_function/main.py
# Per-listing scraper: saves all visible text from each car listing page.
# Also preserves the first Craigslist image URL when available.

import os
import io
import time
import datetime as dt
import requests
import re
import csv
from typing import List

from bs4 import BeautifulSoup
from google.cloud import storage
from flask import Request, jsonify

# ---- Config (overridable via env vars in deploy.yml) ----
BUCKET_NAME = os.environ["BUCKET_NAME"]
BASE_SITE = os.environ.get("BASE_SITE", "https://newhaven.craigslist.org")
SEARCH_PATH = os.environ.get("SEARCH_PATH", "/search/cta")          # cars+trucks
MAX_PAGES = int(os.environ.get("MAX_PAGES", "1"))                   # search pages to scan
MAX_ITEMS_PER_RUN = int(os.environ.get("MAX_ITEMS_PER_RUN", "50"))  # safety cap per run
DELAY_SECS = float(os.environ.get("DELAY_SECS", "1.0"))             # polite delay between requests
USER_AGENT = os.environ.get("USER_AGENT", "UConn-OPIM-Student-Scraper/1.0")

HDRS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.8"
}

POST_PAGE_RE = re.compile(r"/(\d+)\.html?$")
POST_ID_RE = re.compile(r"/(\d+)\.html?$")
IMAGE_URL_RE = re.compile(r"https://images\.craigslist\.org/[^\s\"'<>]+", re.I)


# -- Helpers -------------------------------------------------------------------

def _page_url(base: str, path: str, page: int) -> str:
    # Craigslist uses s=<offset> with 120 results/page.
    # hasPic=1 asks Craigslist to return listings with pictures.
    if page == 0:
        return f"{base}{path}?hasPic=1&srchType=T"
    return f"{base}{path}?hasPic=1&srchType=T&s={page * 120}"


def _extract_listing_links(html: str) -> list[str]:
    """
    Return absolute URLs to individual listings from a search results page.
    Handles classic/new layouts and falls back to regex if needed.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    # Classic layout
    for a in soup.select("a.result-title, a.result-title.hdrlnk"):
        href = a.get("href")
        if href:
            links.add(href)

    # Newer layout
    for a in soup.select("li.cl-search-result a.titlestring"):
        href = a.get("href")
        if href:
            links.add(href)

    # Fallback: any anchor that looks like a posting
    for a in soup.select("li.cl-search-result a, .result-row a, a[href$='.html']"):
        href = a.get("href")
        if href and POST_PAGE_RE.search(href):
            links.add(href)

    # Final fallback: regex scan of raw HTML
    # Matches absolute or relative post URLs ending with /<post_id>.html.
    for m in re.findall(r'href="([^"]+?/\d+\.html)"', html):
        links.add(m)

    # Normalize to absolute URLs
    abs_links = []
    for href in links:
        if href.startswith("//"):
            abs_links.append(f"https:{href}")
        elif href.startswith("/"):
            abs_links.append(f"{BASE_SITE}{href}")
        else:
            abs_links.append(href)

    # Keep only post pages
    abs_links = [u for u in abs_links if POST_PAGE_RE.search(u)]

    return abs_links


def _post_id_from_url(url: str) -> str:
    m = POST_ID_RE.search(url)
    return m.group(1) if m else ""


def _visible_text_from_html(html: str) -> str:
    """
    Extract visible listing text from the Craigslist HTML.
    This intentionally removes script/style/template content.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    raw = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln and not ln.isspace()]

    dedup = []
    for ln in lines:
        if not dedup or ln != dedup[-1]:
            dedup.append(ln)

    return "\n".join(dedup) + "\n"


def _first_image_url_from_html(html: str) -> str:
    """
    Extract the first Craigslist image URL from a listing page HTML.

    The scraper still saves visible text, but visible text alone strips image tags.
    This helper preserves the first image URL before the HTML is converted to text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Common image tags
    for img in soup.find_all("img"):
        for attr in ["src", "data-src"]:
            src = img.get(attr)
            if src and "images.craigslist.org" in src:
                return src

    # Fallback: scan raw HTML for Craigslist image URLs
    m = IMAGE_URL_RE.search(html)
    if m:
        return m.group(0)

    return ""


def _upload_text(bucket: str, object_name: str, text: str):
    storage.Client().bucket(bucket).blob(object_name).upload_from_string(
        text,
        content_type="text/plain"
    )


def _upload_csv(bucket: str, object_name: str, rows: List[dict], header: List[str]):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    w.writerows(rows)

    storage.Client().bucket(bucket).blob(object_name).upload_from_string(
        buf.getvalue(),
        content_type="text/csv"
    )


# -- Entry point ----------------------------------------------------------------

def entrypoint(request: Request):
    """
    HTTP GET.

    Optional query overrides:
      ?pages=2&max=40&base=https://hartford.craigslist.org&path=/search/cta
    """
    pages = min(MAX_PAGES, int(request.args.get("pages", MAX_PAGES)))
    max_items = min(MAX_ITEMS_PER_RUN, int(request.args.get("max", MAX_ITEMS_PER_RUN)))
    base = request.args.get("base", BASE_SITE)
    path = request.args.get("path", SEARCH_PATH)

    # 1) Build run folder: YYYYMMDDHHMMSS UTC
    run_id = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_prefix = f"scrapes/{run_id}"

    # 2) Collect listing links from search pages
    listing_urls = []
    for p in range(pages):
        url = _page_url(base, path, p)

        r = requests.get(url, headers=HDRS, timeout=25)
        r.raise_for_status()

        listing_urls.extend(_extract_listing_links(r.text))

        if p < pages - 1:
            time.sleep(DELAY_SECS)

    # 3) Deduplicate and cap to max_items
    seen = set()
    urls = []

    for u in listing_urls:
        pid = _post_id_from_url(u)

        if pid and pid not in seen:
            seen.add(pid)
            urls.append((pid, u))

        if len(urls) >= max_items:
            break

    # 4) Fetch each listing page and write one TXT per listing
    index_rows = []

    for i, (pid, u) in enumerate(urls, start=1):
        try:
            r = requests.get(u, headers=HDRS, timeout=25)
            r.raise_for_status()

            first_image_url = _first_image_url_from_html(r.text)
            text = _visible_text_from_html(r.text)

            # Preserve image URL in the raw text file so the downstream extractor can parse it.
            if first_image_url:
                text += f"\nimage_url: {first_image_url}\n"

            obj = f"{run_prefix}/{pid}.txt"

            _upload_text(BUCKET_NAME, obj, text)

            index_rows.append({
                "post_id": pid,
                "url": u,
                "object": obj,
                "image_url": first_image_url
            })

            if i < len(urls):
                time.sleep(DELAY_SECS)

        except Exception as e:
            # Record failure in index for transparency
            index_rows.append({
                "post_id": pid,
                "url": u,
                "object": "",
                "image_url": "",
                "error": str(e)
            })

    # 5) Write an optional index.csv for the run
    if index_rows:
        header = sorted({key for row in index_rows for key in row.keys()})
        _upload_csv(BUCKET_NAME, f"{run_prefix}/index.csv", index_rows, header)

    return jsonify({
        "ok": True,
        "run_id": run_id,
        "pages_scanned": pages,
        "candidates_found": len(listing_urls),
        "items_attempted": len(urls),
        "saved_prefix": run_prefix
    })
