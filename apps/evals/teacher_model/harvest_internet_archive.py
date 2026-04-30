"""
Internet Archive public-domain piano pedagogy book harvester for the CPT corpus.

Searches Internet Archive for pre-1928 piano books freely downloadable as PDFs,
scores relevance, and saves to corpus with provenance tracking.

Rate limit: polite crawl, sleep 2s between downloads.

CLI usage:
    uv run python -m teacher_model.harvest_internet_archive
    uv run python -m teacher_model.harvest_internet_archive --dry-run
    uv run python -m teacher_model.harvest_internet_archive --no-filter
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

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

logger = logging.getLogger(__name__)

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"
IA_DOWNLOAD_BASE = "https://archive.org/download"
CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
MIN_WORD_COUNT = 500
DOWNLOAD_SLEEP = 2.0

SEARCH_QUERIES = [
    "piano technique instruction",
    "pianoforte playing method",
    "piano pedagogy teaching",
    "piano practice method",
    "piano interpretation expression",
    "Matthay piano",
    "Leschetizky piano",
    "Czerny piano method",
    "Kullak piano aesthetics",
    "Breithaupt piano natural technic",
]


def _already_in_corpus(url: str) -> bool:
    h = hashlib.sha256(url.encode()).hexdigest()[:12]
    return (CORPUS_DIR / f"pdf_{h}.txt").exists() or (CORPUS_DIR / f"web_{h}.txt").exists()


_USABLE_PDF_FORMATS = {"Text PDF", "Image Container PDF", "Additional Text PDF"}


def _ia_search(query: str, page: int = 1, rows: int = 50) -> list[dict]:
    """Search Internet Archive and return items that have a usable (non-DRM) PDF format."""
    full_query = f"({query}) AND mediatype:texts"
    resp = requests.get(
        IA_SEARCH_URL,
        params={
            "q": full_query,
            "fl[]": ["identifier", "title", "format", "date"],
            "rows": rows,
            "page": page,
            "output": "json",
        },
        headers={"User-Agent": "CrescendAI-Research/1.0 (piano pedagogy corpus collection)"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    docs = data.get("response", {}).get("docs", [])
    return [d for d in docs if _USABLE_PDF_FORMATS & set(d.get("format") or [])]


def _find_pdf_file(identifier: str) -> Optional[str]:
    """
    Query the IA files API to find the best PDF file for an item.

    Prefers the single-file PDF over page-image bundles.
    """
    try:
        resp = requests.get(
            f"https://archive.org/metadata/{identifier}/files",
            headers={"User-Agent": "CrescendAI-Research/1.0"},
            timeout=20,
        )
        resp.raise_for_status()
        files = resp.json().get("result", [])
        # Prefer files ending in .pdf (not _text.pdf which is often OCR noise)
        pdf_files = [
            f["name"] for f in files
            if f.get("name", "").endswith(".pdf")
            and "_encrypted" not in f.get("name", "")
            and "_text" not in f.get("name", "")
            and "_chocr" not in f.get("name", "")
        ]
        if not pdf_files:
            return None
        # Prefer the shortest name (usually the main book PDF)
        pdf_files.sort(key=len)
        return pdf_files[0]
    except Exception as exc:
        logger.debug("IA files API error for %s: %s", identifier, exc)
        return None


def _download_pdf(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "CrescendAI-Research/1.0 (piano pedagogy corpus collection)"},
            timeout=120,
            allow_redirects=True,
        )
        if resp.status_code == 200:
            ct = resp.headers.get("Content-Type", "").lower()
            if "pdf" in ct or resp.content[:4] == b"%PDF":
                return resp.content
        logger.debug("PDF download failed: status=%d url=%s", resp.status_code, url)
        return None
    except Exception as exc:
        logger.debug("PDF download error: %s url=%s", exc, url)
        return None


def harvest(
    max_pages: int = 20,
    rows_per_page: int = 50,
    dry_run: bool = False,
    no_filter: bool = False,
    manifest_path: Optional[Path] = None,
) -> dict:
    """
    Harvest public-domain piano pedagogy books from Internet Archive.

    Returns summary dict: {saved, skipped, already_done, failed, total_checked}.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ProvenanceManifest(path=manifest_path)

    classifier: Optional[PedagogyRelevanceClassifier] = None
    if not no_filter:
        logger.info("Loading relevance classifier...")
        classifier = PedagogyRelevanceClassifier()
        logger.info("Classifier ready (threshold=%.3f)", classifier._threshold)

    saved = skipped = already_done = failed = total_checked = 0
    seen_identifiers: set[str] = set()

    for query in SEARCH_QUERIES:
        logger.info("Query: %r", query)

        for page in range(1, max_pages + 1):
            try:
                items = _ia_search(query, page=page, rows=rows_per_page)
            except Exception as exc:
                logger.warning("IA search error (query=%r page=%d): %s", query, page, exc)
                break

            if not items:
                break

            for item in items:
                identifier = item.get("identifier", "")
                if not identifier or identifier in seen_identifiers:
                    continue
                seen_identifiers.add(identifier)

                total_checked += 1
                title = item.get("title", identifier)

                # Find the actual PDF filename via the files metadata API
                pdf_filename = _find_pdf_file(identifier)
                if not pdf_filename:
                    logger.debug("No usable PDF file for: %s", title[:60])
                    skipped += 1
                    continue

                pdf_url = f"{IA_DOWNLOAD_BASE}/{identifier}/{pdf_filename}"

                if _already_in_corpus(pdf_url):
                    logger.info("Already in corpus: %s", title[:60])
                    already_done += 1
                    continue

                if dry_run:
                    logger.info("[DRY RUN] Would download: %s -> %s", title[:50], pdf_url)
                    saved += 1
                    continue

                logger.info("Downloading: %s", title[:60])
                time.sleep(DOWNLOAD_SLEEP)
                pdf_bytes = _download_pdf(pdf_url)
                if pdf_bytes is None:
                    logger.warning("Download failed: %s", pdf_url[:80])
                    failed += 1
                    continue

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = Path(tmp.name)

                try:
                    result = process_pdf(
                        tmp_path,
                        manifest,
                        classifier=classifier,
                        source_tier="tier2_literature",
                        publisher=f"Internet Archive ({identifier})",
                    )
                    if result["included"] and result["word_count"] >= MIN_WORD_COUNT:
                        h = hashlib.sha256(pdf_url.encode()).hexdigest()[:12]
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

    return {
        "saved": saved,
        "skipped": skipped,
        "already_done": already_done,
        "failed": failed,
        "total_checked": total_checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest public-domain piano books from Internet Archive into the CPT corpus."
    )
    parser.add_argument("--max-pages", type=int, default=20,
                        help="Max search pages per query (default: 20)")
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
    print(f"  saved        = {result['saved']}")
    print(f"  skipped      = {result['skipped']} (no PDF or below threshold)")
    print(f"  already_done = {result['already_done']}")
    print(f"  failed       = {result['failed']}")
    print(f"  total_checked= {result['total_checked']}")


if __name__ == "__main__":
    main()
