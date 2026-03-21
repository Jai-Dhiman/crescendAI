"""Review dashboard for T5 Skill Corpus candidate curation.

Generates an HTML page with embedded YouTube players and skill bucket
selectors. Saves curated results as manifest.yaml files.

Usage:
    # Generate review HTML for all uncurated pieces
    python -m src.data_collection.review_candidates

    # Generate for a specific piece
    python -m src.data_collection.review_candidates --piece moonlight_sonata_mvt1

    # Serve locally (open in browser, curate, save)
    python -m src.data_collection.review_candidates --serve
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
from pathlib import Path

import yaml

SKILL_EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "evals" / "skill_eval"

SKILL_LABELS = {
    1: "Beginner",
    2: "Early Intermediate",
    3: "Intermediate",
    4: "Advanced",
    5: "Professional",
}


# ---------------------------------------------------------------------------
# Load candidates
# ---------------------------------------------------------------------------


def load_all_candidates(base_dir: Path, piece_filter: str | None = None) -> dict:
    """Load all candidates.yaml files. Returns {piece_slug: data_dict}."""
    pieces = {}
    for candidates_file in sorted(base_dir.glob("*/candidates.yaml")):
        piece_slug = candidates_file.parent.name
        if piece_filter and piece_slug != piece_filter:
            continue
        with open(candidates_file) as f:
            data = yaml.safe_load(f)
        if data and data.get("recordings"):
            pieces[piece_slug] = data
    return pieces


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(pieces: dict, base_dir: Path) -> str:
    """Generate the full review dashboard HTML."""
    total_recordings = sum(len(p["recordings"]) for p in pieces.values())

    piece_tabs_html = []
    piece_panels_html = []

    for idx, (slug, data) in enumerate(pieces.items()):
        recordings = data["recordings"]
        n_curated = sum(1 for r in recordings if r.get("skill_bucket") is not None)
        n_skipped = sum(1 for r in recordings if r.get("label_rationale") == "skip")
        n_total = len(recordings)

        # Tab
        active = "active" if idx == 0 else ""
        piece_tabs_html.append(
            f'<button class="tab {active}" onclick="showPiece(\'{slug}\')" '
            f'id="tab-{slug}">'
            f"{data.get('title', slug)} "
            f'<span class="badge">{n_curated}/{n_total}</span>'
            f"</button>"
        )

        # Panel
        cards_html = []
        for ri, rec in enumerate(recordings):
            vid = rec["video_id"]
            bucket_val = rec.get("skill_bucket")
            rationale = rec.get("label_rationale", "") or ""
            is_skipped = rationale == "skip"

            bucket_buttons = []
            for b in range(1, 6):
                selected = "selected" if bucket_val == b else ""
                bucket_buttons.append(
                    f'<button class="bucket-btn {selected}" '
                    f"onclick=\"setBucket('{slug}', {ri}, {b})\">"
                    f"{b}"
                    f"</button>"
                )

            skip_class = "selected" if is_skipped else ""
            card_class = "skipped" if is_skipped else ("curated" if bucket_val else "")

            cards_html.append(f"""
            <div class="card {card_class}" id="card-{slug}-{ri}">
                <div class="card-header">
                    <span class="card-index">#{ri + 1}</span>
                    <span class="card-title">{_escape_html(rec["title"])}</span>
                    <span class="card-channel">{_escape_html(rec["channel"])}</span>
                </div>
                <div class="card-body">
                    <div class="player-container">
                        <iframe
                            width="480" height="270"
                            src="https://www.youtube.com/embed/{vid}"
                            frameborder="0"
                            allowfullscreen
                            loading="lazy"
                        ></iframe>
                    </div>
                    <div class="card-meta">
                        <span>Duration: {rec.get("duration_seconds", "?")}s</span>
                        <span>Views: {_format_views(rec.get("view_count"))}</span>
                    </div>
                    <div class="bucket-selector">
                        <span class="bucket-label">Skill:</span>
                        {"".join(bucket_buttons)}
                        <button class="skip-btn {skip_class}"
                            onclick="skipRecording('{slug}', {ri})">
                            Skip
                        </button>
                    </div>
                </div>
            </div>
            """)

        display = "block" if idx == 0 else "none"
        piece_panels_html.append(
            f'<div class="piece-panel" id="panel-{slug}" style="display:{display}">'
            f'<div class="piece-header">'
            f"<h2>{data.get('title', slug)}</h2>"
            f'<span class="piece-meta">{data.get("composer", "")} '
            f"| {n_total} candidates | {n_curated} curated | {n_skipped} skipped</span>"
            f"</div>"
            f'<div class="cards">{"".join(cards_html)}</div>'
            f"</div>"
        )

    # Build the initial state JSON
    state = {}
    for slug, data in pieces.items():
        state[slug] = {
            "title": data.get("title", slug),
            "composer": data.get("composer", ""),
            "recordings": [
                {
                    "video_id": r["video_id"],
                    "title": r["title"],
                    "channel": r["channel"],
                    "duration_seconds": r.get("duration_seconds"),
                    "view_count": r.get("view_count"),
                    "upload_date": r.get("upload_date"),
                    "skill_bucket": r.get("skill_bucket"),
                    "label_rationale": r.get("label_rationale"),
                    "downloaded": r.get("downloaded", False),
                    "download_error": r.get("download_error"),
                }
                for r in data["recordings"]
            ],
        }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>T5 Skill Corpus - Candidate Review</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: #0a0a0a;
    color: #e0e0e0;
    padding: 20px;
}}
h1 {{
    font-size: 1.4rem;
    margin-bottom: 4px;
    color: #fff;
}}
.subtitle {{
    color: #888;
    font-size: 0.85rem;
    margin-bottom: 16px;
}}
.tabs {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 16px;
    border-bottom: 1px solid #333;
    padding-bottom: 8px;
}}
.tab {{
    background: #1a1a1a;
    border: 1px solid #333;
    color: #aaa;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 0.8rem;
    border-radius: 4px;
    font-family: inherit;
}}
.tab.active {{
    background: #2a2a2a;
    color: #fff;
    border-color: #555;
}}
.badge {{
    background: #333;
    color: #888;
    padding: 1px 6px;
    border-radius: 10px;
    font-size: 0.7rem;
    margin-left: 4px;
}}
.piece-header {{
    margin-bottom: 16px;
}}
.piece-header h2 {{
    font-size: 1.2rem;
    color: #fff;
}}
.piece-meta {{
    color: #888;
    font-size: 0.8rem;
}}
.cards {{
    display: flex;
    flex-direction: column;
    gap: 12px;
}}
.card {{
    background: #151515;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 12px;
}}
.card.curated {{
    border-color: #2d5a2d;
}}
.card.skipped {{
    opacity: 0.4;
    border-color: #5a2d2d;
}}
.card-header {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 8px;
}}
.card-index {{
    color: #555;
    font-size: 0.75rem;
    min-width: 30px;
}}
.card-title {{
    color: #ddd;
    font-size: 0.9rem;
    flex: 1;
}}
.card-channel {{
    color: #777;
    font-size: 0.8rem;
}}
.card-body {{
    display: flex;
    align-items: flex-start;
    gap: 16px;
    flex-wrap: wrap;
}}
.player-container {{
    flex-shrink: 0;
}}
.player-container iframe {{
    border-radius: 4px;
}}
.card-meta {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    color: #888;
    font-size: 0.8rem;
    min-width: 120px;
}}
.bucket-selector {{
    display: flex;
    align-items: center;
    gap: 4px;
    margin-top: 4px;
}}
.bucket-label {{
    color: #888;
    font-size: 0.8rem;
    margin-right: 4px;
}}
.bucket-btn {{
    width: 32px;
    height: 32px;
    border: 1px solid #444;
    background: #1a1a1a;
    color: #aaa;
    cursor: pointer;
    border-radius: 4px;
    font-size: 0.85rem;
    font-family: inherit;
}}
.bucket-btn:hover {{
    background: #2a2a2a;
    color: #fff;
}}
.bucket-btn.selected {{
    background: #1a4a1a;
    color: #4ade80;
    border-color: #4ade80;
}}
.skip-btn {{
    padding: 4px 12px;
    height: 32px;
    border: 1px solid #444;
    background: #1a1a1a;
    color: #aaa;
    cursor: pointer;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-left: 8px;
    font-family: inherit;
}}
.skip-btn:hover {{
    background: #3a1a1a;
    color: #f87171;
}}
.skip-btn.selected {{
    background: #3a1a1a;
    color: #f87171;
    border-color: #f87171;
}}
.save-bar {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #111;
    border-top: 1px solid #333;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 100;
}}
.save-btn {{
    padding: 8px 24px;
    background: #1a4a1a;
    color: #4ade80;
    border: 1px solid #4ade80;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
    font-family: inherit;
}}
.save-btn:hover {{
    background: #2a5a2a;
}}
.save-status {{
    color: #888;
    font-size: 0.85rem;
}}
.progress-summary {{
    color: #aaa;
    font-size: 0.85rem;
}}
</style>
</head>
<body>
<h1>T5 Skill Corpus -- Candidate Review</h1>
<p class="subtitle">{total_recordings} candidates across {len(pieces)} pieces</p>

<div class="tabs">
{"".join(piece_tabs_html)}
</div>

{"".join(piece_panels_html)}

<div class="save-bar">
    <span class="progress-summary" id="progress-summary"></span>
    <span class="save-status" id="save-status"></span>
    <button class="save-btn" onclick="saveAll()">Save All</button>
</div>

<div style="height: 60px;"></div>

<script>
const STATE = {json.dumps(state)};

function showPiece(slug) {{
    document.querySelectorAll('.piece-panel').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('panel-' + slug).style.display = 'block';
    document.getElementById('tab-' + slug).classList.add('active');
}}

function setBucket(slug, idx, bucket) {{
    const rec = STATE[slug].recordings[idx];
    // Toggle off if same bucket clicked
    if (rec.skill_bucket === bucket) {{
        rec.skill_bucket = null;
        rec.label_rationale = null;
    }} else {{
        rec.skill_bucket = bucket;
        rec.label_rationale = 'human curation';
        // Un-skip if was skipped
    }}
    updateCardUI(slug, idx);
    updateProgress();
}}

function skipRecording(slug, idx) {{
    const rec = STATE[slug].recordings[idx];
    if (rec.label_rationale === 'skip') {{
        rec.label_rationale = null;
        rec.skill_bucket = null;
    }} else {{
        rec.label_rationale = 'skip';
        rec.skill_bucket = null;
    }}
    updateCardUI(slug, idx);
    updateProgress();
}}

function updateCardUI(slug, idx) {{
    const rec = STATE[slug].recordings[idx];
    const card = document.getElementById('card-' + slug + '-' + idx);
    const isSkipped = rec.label_rationale === 'skip';
    const isCurated = rec.skill_bucket !== null;

    card.className = 'card' + (isSkipped ? ' skipped' : (isCurated ? ' curated' : ''));

    // Update bucket buttons
    const btns = card.querySelectorAll('.bucket-btn');
    btns.forEach((btn, i) => {{
        btn.className = 'bucket-btn' + (rec.skill_bucket === (i + 1) ? ' selected' : '');
    }});

    // Update skip button
    const skipBtn = card.querySelector('.skip-btn');
    skipBtn.className = 'skip-btn' + (isSkipped ? ' selected' : '');

    // Update tab badge
    const recs = STATE[slug].recordings;
    const nCurated = recs.filter(r => r.skill_bucket !== null).length;
    const badge = document.getElementById('tab-' + slug).querySelector('.badge');
    badge.textContent = nCurated + '/' + recs.length;
}}

function updateProgress() {{
    let totalCurated = 0;
    let totalSkipped = 0;
    let totalRecs = 0;
    for (const slug in STATE) {{
        const recs = STATE[slug].recordings;
        totalRecs += recs.length;
        totalCurated += recs.filter(r => r.skill_bucket !== null).length;
        totalSkipped += recs.filter(r => r.label_rationale === 'skip').length;
    }}
    document.getElementById('progress-summary').textContent =
        totalCurated + ' curated, ' + totalSkipped + ' skipped, ' +
        (totalRecs - totalCurated - totalSkipped) + ' remaining';
}}

function saveAll() {{
    const status = document.getElementById('save-status');
    status.textContent = 'Saving...';

    fetch('/save', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(STATE),
    }})
    .then(r => {{
        if (!r.ok) throw new Error('Save failed: ' + r.status);
        return r.json();
    }})
    .then(data => {{
        status.textContent = 'Saved ' + data.saved + ' manifest(s) at ' +
            new Date().toLocaleTimeString();
    }})
    .catch(err => {{
        status.textContent = 'Error: ' + err.message;
    }});
}}

// Initialize progress
updateProgress();
</script>
</body>
</html>"""


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _format_views(count: int | None) -> str:
    if count is None:
        return "N/A"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)


# ---------------------------------------------------------------------------
# Save handler -- writes manifest.yaml from curated state
# ---------------------------------------------------------------------------


def save_curated_manifests(state: dict, base_dir: Path) -> int:
    """Write labels back into candidates.yaml (persistence) and manifest.yaml (curated subset)."""
    saved = 0
    for slug, data in state.items():
        piece_dir = base_dir / slug
        candidates_path = piece_dir / "candidates.yaml"

        # Update candidates.yaml in-place with labels for persistence/resuming
        if candidates_path.exists():
            with open(candidates_path) as f:
                candidates_data = yaml.safe_load(f)

            if candidates_data and candidates_data.get("recordings"):
                # Build lookup from state by video_id
                state_by_vid = {
                    r["video_id"]: r for r in data["recordings"]
                }

                for rec in candidates_data["recordings"]:
                    vid = rec["video_id"]
                    if vid in state_by_vid:
                        rec["skill_bucket"] = state_by_vid[vid].get("skill_bucket")
                        rec["label_rationale"] = state_by_vid[vid].get("label_rationale")

                # Update status based on labeling progress
                n_labeled = sum(
                    1
                    for r in candidates_data["recordings"]
                    if r.get("skill_bucket") is not None
                    or r.get("label_rationale") == "skip"
                )
                n_total = len(candidates_data["recordings"])
                if n_labeled == n_total:
                    candidates_data["status"] = "complete"
                elif n_labeled > 0:
                    candidates_data["status"] = "partial"

                with open(candidates_path, "w") as f:
                    yaml.dump(
                        candidates_data,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                    )

        # Also write manifest.yaml with only curated recordings
        curated = [r for r in data["recordings"] if r.get("skill_bucket") is not None]
        if not curated:
            saved += 1
            continue

        manifest = {
            "piece": slug,
            "title": data.get("title", slug),
            "composer": data.get("composer", ""),
            "recordings": [
                {
                    "video_id": r["video_id"],
                    "title": r["title"],
                    "channel": r["channel"],
                    "duration_seconds": r.get("duration_seconds"),
                    "skill_bucket": r["skill_bucket"],
                    "label_rationale": r.get("label_rationale", "human curation"),
                    "downloaded": r.get("downloaded", False),
                    "download_error": r.get("download_error"),
                }
                for r in curated
            ],
        }

        piece_dir.mkdir(parents=True, exist_ok=True)
        out_path = piece_dir / "manifest.yaml"
        with open(out_path, "w") as f:
            yaml.dump(
                manifest,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        saved += 1
        print(f"  Saved {slug}: {len(curated)} curated, candidates.yaml updated")

    return saved


# ---------------------------------------------------------------------------
# Local HTTP server for review
# ---------------------------------------------------------------------------


class ReviewHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP handler that serves the review dashboard and handles saves."""

    base_dir: Path  # set by factory
    pieces: dict  # set by factory

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = generate_html(self.pieces, self.base_dir)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/save":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                state = json.loads(body)
                saved = save_curated_manifests(state, self.base_dir)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"saved": saved}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging noise
        pass


def make_handler(base_dir: Path, pieces: dict):
    """Create a handler class with base_dir and pieces bound."""

    class BoundHandler(ReviewHandler):
        pass

    BoundHandler.base_dir = base_dir
    BoundHandler.pieces = pieces
    return BoundHandler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="T5 Skill Corpus candidate review dashboard"
    )
    parser.add_argument("--piece", type=str, help="Filter to a single piece")
    parser.add_argument("--serve", action="store_true", help="Start local HTTP server")
    parser.add_argument(
        "--port", type=int, default=8765, help="Server port (default: 8765)"
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Write static HTML file instead of serving",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=SKILL_EVAL_DIR,
        help="Base directory for skill eval data",
    )

    args = parser.parse_args()

    pieces = load_all_candidates(args.base_dir, piece_filter=args.piece)
    if not pieces:
        print("No candidates.yaml files found.", file=sys.stderr)
        sys.exit(1)

    total = sum(len(p["recordings"]) for p in pieces.values())
    n_curated = sum(
        1
        for p in pieces.values()
        for r in p["recordings"]
        if r.get("skill_bucket") is not None
    )
    n_skipped = sum(
        1
        for p in pieces.values()
        for r in p["recordings"]
        if r.get("label_rationale") == "skip"
    )
    n_remaining = total - n_curated - n_skipped
    print(f"Loaded {total} candidates across {len(pieces)} pieces")
    print(f"  {n_curated} curated, {n_skipped} skipped, {n_remaining} remaining")

    if args.output_html:
        html = generate_html(pieces, args.base_dir)
        args.output_html.write_text(html, encoding="utf-8")
        print(f"Wrote {args.output_html}")
    elif args.serve:
        handler = make_handler(args.base_dir, pieces)
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            print(f"Review dashboard: http://localhost:{args.port}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
    else:
        # Default: print summary and generate static HTML
        html = generate_html(pieces, args.base_dir)
        out_path = args.base_dir / "review.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"Wrote {out_path}")
        print(f"Open in browser: file://{out_path}")
        print()
        print("Or run with --serve for live curation with save support:")
        print(f"  python -m src.data_collection.review_candidates --serve")


if __name__ == "__main__":
    main()
