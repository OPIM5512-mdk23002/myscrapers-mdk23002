# main.py
# Build a single, ever-growing CSV from all structured JSONL files.
# Reads:  gs://<bucket>/<STRUCTURED_PREFIX>/run_id=*/jsonl_llm/*.jsonl
# Writes: gs://<bucket>/<STRUCTURED_PREFIX>/datasets/listings_llm.csv  (atomic publish)
### modified from materialize-v2/main.py to implement 4 new fields: condition, color, body_type, title_status


import csv
import io
import json
import logging
logging.basicConfig(level=logging.INFO)
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME        = os.getenv("GCS_BUCKET")                      # REQUIRED
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured") # e.g., "structured"

storage_client = storage.Client()

# Accept BOTH runIDs:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

# CSV schema
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",

    # Structured modeling fields
    "price", "year", "make", "model", "mileage",
    "transmission", "drivetrain", "fuel_type", "engine_cylinders",
    "condition", "color", "body_type", "title_status",
    "city", "state", "zip_code",

    # Image modeling field
    "image_url",

    # Traceability field
    "source_txt",

    # Text modeling fields
    "combined_text",
    "combined_text_len",
    "has_combined_text",
]

def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:  # populate it.prefixes
        pass
    run_ids = []
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]           # e.g. run_id=20251026170002
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                run_ids.append(rid)
    return sorted(run_ids)

def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    """Yield dict records from .jsonl under .../run_id=<run_id>/jsonl_llm/ (one JSON per file)."""
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/"
    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        data = blob.download_as_text()
        line = data.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            # ensure required keys exist
            rec.setdefault("run_id", run_id)
            yield rec
        except Exception:
            continue

def _run_id_to_dt(rid: str) -> datetime:
    if RUN_ID_ISO_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if RUN_ID_PLAIN_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    # fallback: now
    return datetime.now(timezone.utc)

def _open_gcs_text_writer(bucket: str, key: str):
    """Open a text-mode writer to GCS; close() will finalize the upload."""
    b = storage_client.bucket(bucket)
    blob = b.blob(key)
    # Text mode avoids the flush/finalize pitfall of binary+TextIOWrapper
    return blob.open("w")  # newline handled by csv module

def _parse_gcs_source_txt(source_txt_value: str):
    """
    Convert source_txt into a bucket name and blob name.

    Handles both formats:
    - scrapes/20260429080046/7928717739.txt
    - gs://bucket-name/scrapes/20260429080046/7928717739.txt

    Most current records use the relative blob path format. This helper makes
    the materializer safer if a future pipeline step stores full GCS URIs.
    """
    if not source_txt_value:
        return BUCKET_NAME, None

    s = str(source_txt_value).strip()

    if not s:
        return BUCKET_NAME, None

    if s.startswith("gs://"):
        without_scheme = s.replace("gs://", "", 1)

        if "/" not in without_scheme:
            return BUCKET_NAME, None

        bucket_name, blob_name = without_scheme.split("/", 1)
        return bucket_name, blob_name

    return BUCKET_NAME, s.lstrip("/")

def _download_blob_text_safe(source_txt_value: str) -> str:
    """
    Download raw listing text from GCS.

    This is used as a backfill path for older LLM JSONL records that do not
    already contain combined_text.

    source_txt_value may be either a relative blob path or a full gs:// URI.
    """
    bucket_name, blob_name = _parse_gcs_source_txt(source_txt_value)

    if not blob_name:
        return ""

    try:
        b = storage_client.bucket(bucket_name)
        blob = b.blob(blob_name)
        return blob.download_as_text(timeout=120)
    except Exception as e:
        logging.warning("Could not download source_txt=%s: %s", source_txt_value, e)
        return ""


def _clean_listing_text_for_modeling(raw_text: str) -> str:
    """
    Same cleaning logic used by extractor-llm-poc.

    Keep this synchronized with the extractor version so old and new records
    produce comparable combined_text.
    """
    if not raw_text:
        return ""

    text = str(raw_text)

    text = re.sub(r"(?im)^\s*image_url\s*:\s*.*$", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(
        r"(?im)^\s*(price|asking price|listed price)\s*[:\-]\s*.*$",
        " ",
        text
    )
    text = re.sub(r"\$\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?", " ", text)
    text = re.sub(r"\$\s*\d+(?:\.\d+)?", " ", text)

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _write_csv(records: Iterable[Dict], dest_key: str, columns=CSV_COLUMNS) -> int:
    n = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            row = {c: rec.get(c, None) for c in columns}

            # Backfill combined_text for older LLM JSONL records.
            # New records should already have combined_text from extractor-llm-poc.
            combined_text = row.get("combined_text")

            if not combined_text:
                source_txt_key = row.get("source_txt")
                raw_text = _download_blob_text_safe(source_txt_key)
                combined_text = _clean_listing_text_for_modeling(raw_text)
                row["combined_text"] = combined_text

            row["combined_text_len"] = len(str(row.get("combined_text") or ""))
            row["has_combined_text"] = row["combined_text_len"] > 0

            w.writerow(row)
            n += 1
    return n  # close() finalizes the upload

def materialize_http(request: Request):
    """
    HTTP POST (no body needed).
    Crawls recent structured run folders, de-dupes by post_id (keep newest run),
    and writes one CSV directly to .../datasets/listings_llm.csv.
    Returns JSON with counts and output path.
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not run_ids:
            return jsonify({"ok": False, "error": f"no runs found under {STRUCTURED_PREFIX}/"}), 200
        
        # Limit to most recent N runs to prevent Cloud Function timeout.
        MAX_RUNS = int(os.getenv("MAX_RUNS", "200"))
        if len(run_ids) > MAX_RUNS:
            logging.info("Limiting from %d to %d most recent runs", len(run_ids), MAX_RUNS)
            run_ids = run_ids[-MAX_RUNS:]

        latest_by_post: Dict[str, Dict] = {}
        for rid in run_ids:
            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid):
                pid = rec.get("post_id")
                if not pid:
                    continue
                prev = latest_by_post.get(pid)
                if (prev is None) or (_run_id_to_dt(rec.get("run_id", rid)) > _run_id_to_dt(prev.get("run_id", ""))):
                    latest_by_post[pid] = rec

        base = f"{STRUCTURED_PREFIX}/datasets"
        final_key = f"{base}/listings_llm.csv"

        sorted_records = sorted(
            latest_by_post.values(), 
            key=lambda r: r.get("scraped_at") or "",# sort by scraped_at if available, otherwise no particular order
            )
        rows = _write_csv(sorted_records, final_key)

        return jsonify({
            "ok": True,
            "runs_scanned": len(run_ids),
            "unique_listings": len(latest_by_post),
            "rows_written": rows,
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200
    except Exception as e:
        # Return a JSON error so you don't just see a plain 500
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
