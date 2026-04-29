"""
PDF and web text extraction pipeline for Tier 2-3 CPT corpus collection.

Extracts text from PDFs (PyMuPDF) and web pages (BeautifulSoup), scores
relevance against the pedagogy centroid, and saves to the corpus with full
provenance tracking.

CLI usage:
    uv run python -m teacher_model.scrape_text --pdf path/to/book.pdf
    uv run python -m teacher_model.scrape_text --pdf-dir path/to/pdfs/
    uv run python -m teacher_model.scrape_text --url "https://example.com/article"
    uv run python -m teacher_model.scrape_text --url-list urls.txt
    uv run python -m teacher_model.scrape_text --no-filter
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

from teacher_model.provenance import ProvenanceManifest, ProvenanceRecord
from teacher_model.relevance_classifier import PedagogyRelevanceClassifier

_CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
_USER_AGENT = "CrescendAI-Research/1.0 (piano pedagogy corpus collection)"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract and concatenate text from all pages of a PDF."""
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_web_text(url: str) -> tuple[str, str]:
    """
    Fetch a web page and extract readable article text.

    Returns (text, title).  Strips boilerplate (nav, footer, scripts).
    Tries <article>, then <main>, then <body> for content root.
    Extracts text from p/h1-h4/li tags within the chosen root.
    """
    response = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove boilerplate tags before extraction
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Find the best content root
    root = soup.find("article") or soup.find("main") or soup.find("body")
    if root is None:
        raise ValueError(f"No usable content root found in page: {url}")

    text_parts: list[str] = []
    for elem in root.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        part = elem.get_text(separator=" ", strip=True)
        if part:
            text_parts.append(part)

    text = "\n\n".join(text_parts)
    return text, title


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    # Strip Unicode control characters (except common whitespace)
    cleaned_chars: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\t", "\r"):
            continue
        cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of spaces/tabs on a single line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # Collapse 2+ consecutive blank lines down to 1
    result_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                result_lines.append(line)
        else:
            blank_run = 0
            result_lines.append(line)

    return "\n".join(result_lines).strip()


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------


def _hash_key(key: str) -> str:
    """12-char SHA-256 prefix for stable filenames."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _save_corpus(filename: str, text: str) -> Path:
    _CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CORPUS_DIR / filename
    dest.write_text(text, encoding="utf-8")
    return dest


def process_pdf(
    pdf_path: str | Path,
    manifest: ProvenanceManifest,
    classifier: Optional[PedagogyRelevanceClassifier] = None,
    source_tier: str = "tier2_literature",
    publisher: str = "Unknown",
) -> dict:
    """
    Extract, clean, optionally filter, and save a PDF to the corpus.

    Returns a result dict with keys: path, filename, word_count, score,
    included, corpus_path.
    """
    pdf_path = Path(pdf_path)
    raw = extract_pdf_text(pdf_path)
    text = clean_text(raw)

    score: Optional[float] = None
    included = True
    if classifier is not None:
        score = classifier.score(text[:2000])
        included = classifier.is_relevant(text[:2000])

    word_count = len(text.split())
    h = _hash_key(str(pdf_path))
    filename = f"pdf_{h}.txt"
    corpus_path: Optional[Path] = None

    if included:
        corpus_path = _save_corpus(filename, text)
        record = ProvenanceRecord(
            url=pdf_path.as_uri(),
            title=pdf_path.stem,
            channel_or_publisher=publisher,
            download_timestamp=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            license_claimed="unknown",
            word_count=word_count,
            inclusion_threshold_score=score,
            source_tier=source_tier,
        )
        manifest.add(record)

    return {
        "path": str(pdf_path),
        "filename": filename,
        "word_count": word_count,
        "score": score,
        "included": included,
        "corpus_path": str(corpus_path) if corpus_path else None,
    }


def process_url(
    url: str,
    manifest: ProvenanceManifest,
    classifier: Optional[PedagogyRelevanceClassifier] = None,
    source_tier: str = "tier2_literature",
) -> dict:
    """
    Fetch, clean, optionally filter, and save a URL to the corpus.

    If the URL ends with .pdf, downloads to a tempfile and dispatches to
    process_pdf. Otherwise scrapes as a web page.

    Returns a result dict with keys: url, filename, title, word_count, score,
    included, corpus_path.
    """
    url_path = url.lower().split("?")[0]
    is_pdf_url = url_path.endswith(".pdf")

    if not is_pdf_url:
        # Sniff content-type for URLs that serve PDF without .pdf extension
        head = requests.head(url, headers={"User-Agent": _USER_AGENT}, timeout=15, allow_redirects=True)
        if "application/pdf" in head.headers.get("Content-Type", ""):
            is_pdf_url = True

    if is_pdf_url:
        import tempfile
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=60, allow_redirects=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = Path(tmp.name)
        try:
            result = process_pdf(
                tmp_path,
                manifest,
                classifier=classifier,
                source_tier=source_tier,
                publisher=url,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        # normalise to the same shape process_url returns
        result["url"] = url
        result.setdefault("title", Path(url).stem)
        return result

    raw, title = extract_web_text(url)
    text = clean_text(raw)

    score: Optional[float] = None
    included = True
    if classifier is not None:
        score = classifier.score(text[:2000])
        included = classifier.is_relevant(text[:2000])

    word_count = len(text.split())
    h = _hash_key(url)
    filename = f"web_{h}.txt"
    corpus_path: Optional[Path] = None

    if included:
        corpus_path = _save_corpus(filename, text)
        record = ProvenanceRecord(
            url=url,
            title=title,
            channel_or_publisher="web",
            download_timestamp=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            license_claimed="unknown",
            word_count=word_count,
            inclusion_threshold_score=score,
            source_tier=source_tier,
        )
        manifest.add(record)

    return {
        "url": url,
        "filename": filename,
        "title": title,
        "word_count": word_count,
        "score": score,
        "included": included,
        "corpus_path": str(corpus_path) if corpus_path else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs and web pages into the CPT corpus."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", metavar="PATH", help="Single PDF file to process")
    source.add_argument(
        "--pdf-dir", metavar="DIR", help="Directory of PDFs to process"
    )
    source.add_argument("--url", metavar="URL", help="Single URL to scrape")
    source.add_argument(
        "--url-list",
        metavar="FILE",
        help="Text file with one URL per line",
    )

    parser.add_argument(
        "--tier",
        default="tier2_literature",
        choices=[
            "tier1_youtube",
            "tier2_literature",
            "tier3_musicology",
            "tier4_own",
        ],
        help="Source tier for provenance (default: tier2_literature)",
    )
    parser.add_argument(
        "--publisher",
        default="Unknown",
        help="Publisher name for PDF provenance (default: Unknown)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip relevance filtering; include all documents",
    )
    parser.add_argument(
        "--manifest",
        metavar="PATH",
        help="Path to provenance manifest JSONL (default: teacher_model/data/provenance.jsonl)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    manifest = ProvenanceManifest(path=manifest_path)

    classifier: Optional[PedagogyRelevanceClassifier] = None
    if not args.no_filter:
        print("Loading relevance classifier...")
        classifier = PedagogyRelevanceClassifier()
        print(f"Classifier ready (threshold={classifier._threshold:.3f})")  # type: ignore[union-attr]

    results: list[dict] = []
    failed_urls: list[tuple[str, str]] = []

    if args.pdf:
        result = process_pdf(
            args.pdf,
            manifest,
            classifier=classifier,
            source_tier=args.tier,
            publisher=args.publisher,
        )
        results.append(result)

    elif args.pdf_dir:
        pdf_files = sorted(Path(args.pdf_dir).glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {args.pdf_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(pdf_files)} PDF(s) in {args.pdf_dir}")
        for pdf_file in pdf_files:
            print(f"  Processing {pdf_file.name}...")
            result = process_pdf(
                pdf_file,
                manifest,
                classifier=classifier,
                source_tier=args.tier,
                publisher=args.publisher,
            )
            results.append(result)

    elif args.url:
        result = process_url(
            args.url,
            manifest,
            classifier=classifier,
            source_tier=args.tier,
        )
        results.append(result)

    elif args.url_list:
        url_file = Path(args.url_list)
        urls = [
            line.strip()
            for line in url_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not urls:
            print(f"No URLs found in {args.url_list}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(urls)} URL(s) in {args.url_list}")
        for url in urls:
            print(f"  Fetching {url}...")
            try:
                result = process_url(
                    url,
                    manifest,
                    classifier=classifier,
                    source_tier=args.tier,
                )
                results.append(result)
            except Exception as exc:
                print(f"  ERROR {url}: {exc}", file=sys.stderr)
                failed_urls.append((url, str(exc)))

    # Summary
    included = [r for r in results if r["included"]]
    skipped = [r for r in results if not r["included"]]
    total_words = sum(r["word_count"] for r in included)

    print(f"\nDone: {len(included)} included, {len(skipped)} skipped")
    print(f"Total words saved: {total_words:,}")
    if included:
        print("Saved files:")
        for r in included:
            score_str = f"  score={r['score']:.3f}" if r["score"] is not None else ""
            print(f"  {r['filename']} ({r['word_count']:,} words{score_str})")
    if skipped:
        print("Skipped (below relevance threshold):")
        for r in skipped:
            key = r.get("path") or r.get("url", "?")
            score_str = f"  score={r['score']:.3f}" if r["score"] is not None else ""
            print(f"  {key}{score_str}")
    if failed_urls:
        print(f"Failed ({len(failed_urls)} errors):")
        for url, err in failed_urls:
            print(f"  {url}: {err}")


if __name__ == "__main__":
    main()
