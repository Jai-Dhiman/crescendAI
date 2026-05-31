"""Minimal local HTTP server for the practice_eval review UI.

Serves model/data/evals/practice_eval/review.html and per-piece candidates.yaml
files, and handles POST /update to persist approve/reject decisions.

Usage:
    uv run python scripts/serve_practice_review.py
    uv run python scripts/serve_practice_review.py --port 9000

Then open http://localhost:8765 in your browser.

Dependencies:
    - ruamel.yaml (for round-trip YAML preservation)
    Install: uv add ruamel.yaml   (or it is already in model/pyproject.toml)
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
from pathlib import Path

from ruamel.yaml import YAML

import re

PRACTICE_EVAL_DIR = (Path(__file__).resolve().parents[1] / "data" / "evals" / "practice_eval").resolve()
REVIEW_HTML       = PRACTICE_EVAL_DIR / "review.html"
_PIECE_SLUG_RE    = re.compile(r"[a-z0-9_]+")


def _resolve_piece_yaml(piece: object) -> Path | None:
    """Validate `piece` and return its candidates.yaml path, or None if invalid.

    Two layers: strict allowlist on the slug, then a resolved-path containment
    check so symlinks/normalization quirks cannot escape PRACTICE_EVAL_DIR.
    """
    if not isinstance(piece, str) or not _PIECE_SLUG_RE.fullmatch(piece):
        return None
    candidate = (PRACTICE_EVAL_DIR / piece / "candidates.yaml").resolve()
    if not candidate.is_relative_to(PRACTICE_EVAL_DIR):
        return None
    return candidate

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = False
_yaml.width = 4096  # prevent unwanted line-wrapping


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ReviewHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._serve_file(REVIEW_HTML, "text/html; charset=utf-8")
            return

        # Serve per-piece candidates.yaml:  /data/<piece>/candidates.yaml
        if path.startswith("/data/") and path.endswith("/candidates.yaml"):
            parts = path.lstrip("/").split("/")
            if len(parts) == 3:
                yaml_path = _resolve_piece_yaml(parts[1])
                if yaml_path is not None and yaml_path.exists():
                    self._serve_file(yaml_path, "application/x-yaml; charset=utf-8")
                    return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/update":
            self._handle_update()
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _serve_file(self, path: Path, content_type: str) -> None:
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _json_response(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_update(self) -> None:
        """POST /update -- body: {piece, video_id, approved, review_notes}"""
        try:
            payload = self._read_json_body()
        except (json.JSONDecodeError, ValueError) as e:
            self._json_response(400, {"error": f"bad JSON: {e}"})
            return

        piece     = payload.get("piece")
        video_id  = payload.get("video_id")
        approved  = payload.get("approved")      # true | false | null
        notes     = payload.get("review_notes", "")

        if not piece or not video_id:
            self._json_response(400, {"error": "piece and video_id are required"})
            return

        if approved not in (True, False, None):
            self._json_response(400, {"error": "approved must be true, false, or null"})
            return

        yaml_path = _resolve_piece_yaml(piece)
        if yaml_path is None:
            self._json_response(400, {"error": "invalid piece slug"})
            return
        if not yaml_path.exists():
            self._json_response(404, {"error": f"candidates.yaml not found for piece '{piece}'"})
            return

        try:
            with open(yaml_path) as f:
                data = _yaml.load(f)
        except Exception as e:
            self._json_response(500, {"error": f"failed to load YAML: {e}"})
            return

        if not data or "recordings" not in data:
            self._json_response(500, {"error": "malformed candidates.yaml (no recordings key)"})
            return

        matched = False
        for rec in data["recordings"]:
            if rec.get("video_id") == video_id:
                rec["approved"]      = approved
                rec["review_notes"]  = notes
                matched = True
                break

        if not matched:
            self._json_response(404, {"error": f"video_id '{video_id}' not found in {piece}"})
            return

        try:
            with open(yaml_path, "w") as f:
                _yaml.dump(data, f)
        except Exception as e:
            self._json_response(500, {"error": f"failed to write YAML: {e}"})
            return

        self._json_response(200, {"ok": True, "piece": piece, "video_id": video_id})

    def log_message(self, format: str, *args) -> None:
        print(f"  {self.address_string()} {format % args}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the practice_eval review UI"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    args = parser.parse_args()

    if not REVIEW_HTML.exists():
        raise FileNotFoundError(
            f"review.html not found at {REVIEW_HTML}. "
            "It should be created alongside this tool."
        )

    with socketserver.TCPServer(("127.0.0.1", args.port), ReviewHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Practice review server running at http://localhost:{args.port}")
        print(f"Serving from: {PRACTICE_EVAL_DIR}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
