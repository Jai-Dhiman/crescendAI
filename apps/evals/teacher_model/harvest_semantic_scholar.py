"""
Semantic Scholar open-access paper harvester for the CPT corpus.

Searches Semantic Scholar for piano pedagogy papers, downloads available
open-access PDFs, scores relevance, and saves to corpus with provenance tracking.

Rate limit: 100 requests per 5 minutes (unauthenticated) -> sleep 3.5s between calls.
PDF downloads go to publisher servers (not counted against SS quota).

CLI usage:
    uv run python -m teacher_model.harvest_semantic_scholar
    uv run python -m teacher_model.harvest_semantic_scholar --max-pages 50 --dry-run
    uv run python -m teacher_model.harvest_semantic_scholar --no-filter
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

SS_API_BASE = "https://api.semanticscholar.org/graph/v1"
CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
MIN_WORD_COUNT = 200
API_SLEEP = 20.0

SEARCH_QUERIES = [
    "piano pedagogy teaching",
    "piano performance practice technique",
    "music performance anxiety piano",
    "piano sight reading cognitive",
    "piano memorization performance",
    "music teacher feedback instruction",
    "piano technique biomechanics injury",
    "musical expression phrasing dynamics",
    "piano practice strategies deliberate",
    "keyboard performance historically informed",
    "music performance expertise acquisition",
    "piano voicing tone touch",
    "piano masterclass instruction",
    "music education instrumental teaching",
    "piano interpretation style analysis",
]


def _already_in_corpus(url: str) -> bool:
    h = hashlib.sha256(url.encode()).hexdigest()[:12]
    return (CORPUS_DIR / f"pdf_{h}.txt").exists() or (CORPUS_DIR / f"web_{h}.txt").exists()


def _download_pdf(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "CrescendAI-Research/1.0 (piano pedagogy corpus collection)"},
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


def _ss_search(query: str, offset: int = 0, limit: int = 100) -> dict:
    resp = requests.get(
        f"{SS_API_BASE}/paper/search",
        params={
            "query": query,
            "fields": "title,openAccessPdf,abstract,year,authors",
            "limit": limit,
            "offset": offset,
        },
        headers={"User-Agent": "CrescendAI-Research/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def harvest(
    max_pages: int = 100,
    limit_per_page: int = 100,
    dry_run: bool = False,
    no_filter: bool = False,
    manifest_path: Optional[Path] = None,
) -> dict:
    """
    Harvest piano pedagogy papers from Semantic Scholar.

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
        offset = 0
        pages_this_query = 0

        while pages_this_query < max_pages:
            time.sleep(API_SLEEP)
            try:
                data = _ss_search(query, offset=offset, limit=limit_per_page)
            except Exception as exc:
                logger.warning("SS API error (query=%r offset=%d): %s", query, offset, exc)
                break

            papers = data.get("data", [])
            if not papers:
                break

            for paper in papers:
                paper_id = paper.get("paperId", "")
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                total_checked += 1
                title = paper.get("title", "") or ""
                oa_pdf = paper.get("openAccessPdf") or {}
                pdf_url = oa_pdf.get("url", "") or ""

                if not pdf_url:
                    logger.debug("No open-access PDF for: %s", title[:60])
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
                        publisher="Semantic Scholar",
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

            offset += limit_per_page
            pages_this_query += 1

            total_results = data.get("total", 0)
            if offset >= total_results:
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
        description="Harvest piano pedagogy papers from Semantic Scholar into the CPT corpus."
    )
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Max pages per query (default: 100)")
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
    print(f"  skipped      = {result['skipped']} (no open PDF or below threshold)")
    print(f"  already_done = {result['already_done']}")
    print(f"  failed       = {result['failed']}")
    print(f"  total_checked= {result['total_checked']}")


if __name__ == "__main__":
    main()
