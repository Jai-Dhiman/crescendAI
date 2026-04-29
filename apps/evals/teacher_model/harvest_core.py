"""
CORE.ac.uk dissertation and open-access paper harvester for the CPT corpus.

Searches CORE for piano pedagogy dissertations/theses/papers, downloads available
PDFs, scores relevance, and saves to corpus with provenance tracking.

Rate limit: 10 requests per 10s window (registered key), 1000 requests/day.
PDF downloads go to institutional servers (not CORE quota).

CLI usage:
    uv run python -m teacher_model.harvest_core
    uv run python -m teacher_model.harvest_core --max-pages 20 --dry-run
    uv run python -m teacher_model.harvest_core --no-filter
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

# Load .env
_env_file = Path(__file__).resolve().parents[1] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from teacher_model.provenance import ProvenanceManifest
from teacher_model.relevance_classifier import PedagogyRelevanceClassifier
from teacher_model.scrape_text import process_pdf

import tempfile
import hashlib

logger = logging.getLogger(__name__)

CORE_API_BASE = "https://api.core.ac.uk/v3"
CORPUS_DIR = Path(__file__).parent / "data" / "corpus"

# Targeted search queries for piano pedagogy
SEARCH_QUERIES = [
    "piano pedagogy teaching technique",
    "piano teaching method adult learner",
    "piano masterclass instruction technique",
    "piano technique fingering practice",
    "piano performance anxiety stage fright",
    "piano sight reading practice strategy",
    "piano phrasing dynamics expression",
    "piano memorization practice mental",
    "piano injury prevention technique",
    "piano voicing tone production touch",
]

# Minimum word count to save (very short abstracts not useful)
MIN_WORD_COUNT = 200

# Seconds to sleep between CORE API calls (10 req/min limit -> 12s = safe margin)
API_SLEEP = 12.0


def _core_search(
    query: str,
    key: str,
    offset: int = 0,
    limit: int = 10,
) -> dict:
    """Single paginated CORE search request."""
    resp = requests.get(
        f"{CORE_API_BASE}/search/works",
        params={"q": query, "limit": limit, "offset": offset},
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _already_in_corpus(url: str) -> bool:
    """Check if URL has already been harvested (hash match in corpus)."""
    h = hashlib.sha256(url.encode()).hexdigest()[:12]
    # Both pdf_ and web_ prefixes
    return (CORPUS_DIR / f"pdf_{h}.txt").exists() or (CORPUS_DIR / f"web_{h}.txt").exists()


def _download_pdf(url: str, key: str) -> Optional[bytes]:
    """Download a PDF from a URL. Returns content bytes or None on failure."""
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": "CrescendAI-Research/1.0 (piano pedagogy corpus collection)",
                "Authorization": f"Bearer {key}" if "core.ac.uk" in url else "",
            },
            timeout=60,
            allow_redirects=True,
        )
        if resp.status_code == 200 and "pdf" in resp.headers.get("Content-Type", "").lower():
            return resp.content
        if resp.status_code == 200 and len(resp.content) > 5000:
            # Some servers don't set content-type correctly
            if resp.content[:4] == b"%PDF":
                return resp.content
        logger.debug("PDF download failed: status=%d url=%s", resp.status_code, url)
        return None
    except Exception as exc:
        logger.debug("PDF download error: %s url=%s", exc, url)
        return None


def harvest(
    max_pages: int = 50,
    limit_per_page: int = 10,
    dry_run: bool = False,
    no_filter: bool = False,
    manifest_path: Optional[Path] = None,
) -> dict:
    """
    Harvest piano pedagogy papers from CORE.

    Returns summary dict: {saved, skipped, already_done, failed, total_checked}.
    """
    key = os.environ.get("CORE_API_KEY")
    if not key:
        raise RuntimeError("CORE_API_KEY is not set in environment")

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ProvenanceManifest(path=manifest_path)

    classifier: Optional[PedagogyRelevanceClassifier] = None
    if not no_filter:
        logger.info("Loading relevance classifier...")
        classifier = PedagogyRelevanceClassifier()
        logger.info("Classifier ready (threshold=%.3f)", classifier._threshold)  # type: ignore[union-attr]

    saved = skipped = already_done = failed = total_checked = 0
    seen_ids: set[str] = set()

    for query in SEARCH_QUERIES:
        logger.info("Query: %r", query)
        offset = 0
        pages_this_query = 0

        while pages_this_query < max_pages:
            time.sleep(API_SLEEP)
            try:
                data = _core_search(query, key, offset=offset, limit=limit_per_page)
            except Exception as exc:
                logger.warning("CORE API error (query=%r offset=%d): %s", query, offset, exc)
                break

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                work_id = str(work.get("id", ""))
                if work_id in seen_ids:
                    continue
                seen_ids.add(work_id)

                total_checked += 1
                title = work.get("title", "") or ""
                download_url = work.get("downloadUrl", "") or ""

                if not download_url:
                    logger.debug("No downloadUrl for: %s", title[:60])
                    skipped += 1
                    continue

                if _already_in_corpus(download_url):
                    logger.info("Already in corpus: %s", title[:60])
                    already_done += 1
                    continue

                if dry_run:
                    logger.info("[DRY RUN] Would download: %s", title[:60])
                    saved += 1
                    continue

                logger.info("Downloading: %s", title[:60])
                pdf_bytes = _download_pdf(download_url, key)
                if pdf_bytes is None:
                    logger.warning("Download failed: %s", download_url[:80])
                    failed += 1
                    continue

                # Write to temp file and process through the PDF pipeline
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = Path(tmp.name)

                try:
                    result = process_pdf(
                        tmp_path,
                        manifest,
                        classifier=classifier,
                        source_tier="tier3_musicology",
                        publisher=work.get("publisher", "") or work.get("journals", [{}])[0].get("title", "CORE") if work.get("journals") else "CORE",
                    )
                    if result["included"] and result["word_count"] >= MIN_WORD_COUNT:
                        # Rename corpus file to use download_url hash for dedup
                        h = hashlib.sha256(download_url.encode()).hexdigest()[:12]
                        old_path = Path(result["corpus_path"])
                        new_path = CORPUS_DIR / f"pdf_{h}.txt"
                        if old_path.exists() and not new_path.exists():
                            old_path.rename(new_path)
                        logger.info(
                            "Saved: %s (words=%d score=%.3f)",
                            title[:50],
                            result["word_count"],
                            result["score"] or 0,
                        )
                        saved += 1
                    else:
                        if result["included"] and result["word_count"] < MIN_WORD_COUNT:
                            logger.info("Too short (%d words): %s", result["word_count"], title[:50])
                        skipped += 1
                except Exception as exc:
                    logger.warning("process_pdf failed for %s: %s", title[:50], exc)
                    failed += 1
                finally:
                    tmp_path.unlink(missing_ok=True)

            offset += limit_per_page
            pages_this_query += 1

            if offset >= data.get("totalHits", 0):
                break

    return {
        "saved": saved,
        "skipped": skipped,
        "already_done": already_done,
        "failed": failed,
        "total_checked": total_checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest piano pedagogy papers from CORE.ac.uk into the CPT corpus."
    )
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max CORE pages per query (default: 50)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip relevance filtering, save all documents")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be downloaded without saving")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Provenance manifest path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    result = harvest(
        max_pages=args.max_pages,
        dry_run=args.dry_run,
        no_filter=args.no_filter,
        manifest_path=args.manifest,
    )

    print(f"\nDone:")
    print(f"  saved       = {result['saved']}")
    print(f"  skipped     = {result['skipped']} (no PDF or below relevance threshold)")
    print(f"  already_done= {result['already_done']}")
    print(f"  failed      = {result['failed']}")
    print(f"  total_checked={result['total_checked']}")


if __name__ == "__main__":
    main()
