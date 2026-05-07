"""
OpenAlex open-access academic paper harvester for the CPT corpus.

Searches OpenAlex (250M+ works, no daily quota) for piano pedagogy papers,
downloads available open-access PDFs, scores relevance, and saves to corpus
with provenance tracking.

Rate limit: 10 req/sec unauthenticated; polite pool (100k req/day) with mailto param.
Sleep: 0.15s between API requests.

CLI usage:
    uv run python -m teacher_model.harvest_openalex
    uv run python -m teacher_model.harvest_openalex --dry-run
    uv run python -m teacher_model.harvest_openalex --no-filter
    nohup uv run python -m teacher_model.harvest_openalex --manifest \\
        teacher_model/data/provenance_openalex.jsonl > /tmp/harvest_oa.log 2>&1 &
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

OA_API_BASE = "https://api.openalex.org/works"
OA_MAILTO = "jaidhiman2000@gmail.com"
CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
MIN_WORD_COUNT = 200
API_SLEEP = 0.15

SEARCH_QUERIES = [
    "piano pedagogy teaching technique",
    "piano performance practice",
    "music performance anxiety",
    "piano sight reading",
    "piano memorization practice",
    "music teacher feedback instruction",
    "piano technique injury prevention",
    "musical expression dynamics phrasing",
    "deliberate practice music",
    "piano masterclass instruction",
    "piano voicing tone production",
    "music performance assessment",
    "piano practice strategies",
    "music education instrumental",
    "piano interpretation style",
    "piano adult learner beginner",
    "piano competition performance",
    "music conservatory teaching",
    "piano group lesson teaching",
    "piano fingering hand position",
]


def _already_in_corpus(url: str) -> bool:
    h = hashlib.sha256(url.encode()).hexdigest()[:12]
    return (CORPUS_DIR / f"pdf_{h}.txt").exists() or (CORPUS_DIR / f"web_{h}.txt").exists()


def _download_pdf(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "CrescendAI-Research/1.0 (piano pedagogy corpus; mailto:jaidhiman2000@gmail.com)"},
            timeout=60,
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


def _oa_search(query: str, cursor: str = "*", per_page: int = 200) -> dict:
    """Fetch one page of OpenAlex results. Returns the raw API response dict."""
    resp = requests.get(
        OA_API_BASE,
        params={
            "search": query,
            "filter": "open_access.is_oa:true",
            "per-page": per_page,
            "cursor": cursor,
            "select": "id,display_name,open_access",
            "mailto": OA_MAILTO,
        },
        headers={"User-Agent": "CrescendAI-Research/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def harvest(
    max_pages_per_query: int = 50,
    dry_run: bool = False,
    no_filter: bool = False,
    manifest_path: Optional[Path] = None,
) -> dict:
    """
    Harvest piano pedagogy papers from OpenAlex.

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
    seen_ids: set[str] = set()

    for query in SEARCH_QUERIES:
        logger.info("Query: %r", query)
        cursor: Optional[str] = "*"
        pages_this_query = 0

        while cursor and pages_this_query < max_pages_per_query:
            time.sleep(API_SLEEP)
            try:
                data = _oa_search(query, cursor=cursor)
            except Exception as exc:
                logger.warning("OpenAlex API error (query=%r cursor=%s): %s", query, cursor, exc)
                break

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                work_id = work.get("id", "")
                if work_id in seen_ids:
                    continue
                seen_ids.add(work_id)

                total_checked += 1
                title = work.get("display_name", "") or ""
                oa_info = work.get("open_access") or {}
                pdf_url = oa_info.get("oa_url", "") or ""

                if not pdf_url:
                    logger.debug("No open-access URL for: %s", title[:60])
                    skipped += 1
                    continue

                if _already_in_corpus(pdf_url):
                    logger.info("Already in corpus: %s", title[:60])
                    already_done += 1
                    continue

                if dry_run:
                    logger.info("[DRY RUN] Would download: %s", title[:60])
                    saved += 1
                    continue

                logger.info("Downloading: %s", title[:60])
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
                        source_tier="tier3_musicology",
                        publisher="OpenAlex",
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

            cursor = data.get("meta", {}).get("next_cursor")
            pages_this_query += 1

    return {
        "saved": saved,
        "skipped": skipped,
        "already_done": already_done,
        "failed": failed,
        "total_checked": total_checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest piano pedagogy papers from OpenAlex into the CPT corpus."
    )
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max cursor pages per query (default: 50, each page = 200 results)")
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
        max_pages_per_query=args.max_pages,
        dry_run=args.dry_run,
        no_filter=args.no_filter,
        manifest_path=args.manifest,
    )

    print(f"\nDone:")
    print(f"  saved        = {result['saved']}")
    print(f"  skipped      = {result['skipped']} (no open PDF or below threshold)")
    print(f"  already_done = {result['already_done']}")
    print(f"  failed       = {result['failed']}")
    print(f"  total_checked= {result['total_checked']}")


if __name__ == "__main__":
    main()
