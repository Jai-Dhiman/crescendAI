"""
CrescendAI Eval Viewer -- standalone dev tool.

Run:  python viewer.py
Open:  http://localhost:3333
"""

import http.server
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]
RECORDINGS_JSONL = PROJECT_ROOT / "model" / "data" / "intermediate_cache" / "recordings.jsonl"
AUDIO_DIR = PROJECT_ROOT / "model" / "data" / "eval" / "youtube_amt"
CACHE_DIR = PROJECT_ROOT / "model" / "data" / "eval" / "inference_cache"
TRACES_DIR = PROJECT_ROOT / "model" / "data" / "eval" / "traces"
VIEWER_HTML = Path(__file__).parent / "viewer.html"

PORT = 3333


def _find_cache_subdir():
    """Return the first subdirectory inside CACHE_DIR (the versioned fingerprint dir)."""
    if not CACHE_DIR.is_dir():
        return None
    for entry in sorted(CACHE_DIR.iterdir()):
        if entry.is_dir():
            return entry
    return None


CACHE_SUBDIR = _find_cache_subdir()


def _load_recordings_jsonl():
    """Load recordings.jsonl into a dict keyed by video_id."""
    recordings = {}
    if not RECORDINGS_JSONL.is_file():
        return recordings
    with open(RECORDINGS_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec.get("video_id")
            if vid:
                recordings[vid] = rec
    return recordings


def _count_cache_chunks(video_id):
    """Return chunk count from inference cache, or None if no cache file."""
    if not CACHE_SUBDIR:
        return None
    cache_file = CACHE_SUBDIR / f"{video_id}.json"
    if not cache_file.is_file():
        return None
    with open(cache_file, "r") as f:
        data = json.loads(f.read())
    return data.get("total_chunks", len(data.get("chunks", [])))


def _count_traces(video_id):
    """Count trace files for a given video_id."""
    if not TRACES_DIR.is_dir():
        return 0
    count = 0
    prefix = f"{video_id}_chunk"
    for entry in TRACES_DIR.iterdir():
        if entry.name.startswith(prefix) and entry.suffix == ".json":
            count += 1
    return count


class ViewerHandler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        if not path.is_file():
            self.send_error(404, f"Not found: {path.name}")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query params

        if path == "/":
            self._send_file(VIEWER_HTML, "text/html; charset=utf-8")

        elif path == "/api/recordings":
            self._handle_recordings()

        elif path.startswith("/api/recording/"):
            video_id = path[len("/api/recording/"):]
            self._handle_recording_detail(video_id)

        elif path.startswith("/api/traces/"):
            video_id = path[len("/api/traces/"):]
            self._handle_traces(video_id)

        elif path.startswith("/audio/"):
            filename = path[len("/audio/"):]
            audio_path = AUDIO_DIR / filename
            self._send_file(audio_path, "audio/wav")

        else:
            self.send_error(404, "Not found")

    def _handle_recordings(self):
        recs = _load_recordings_jsonl()
        result = []
        for vid, rec in recs.items():
            chunk_count = _count_cache_chunks(vid)
            obs_count = _count_traces(vid)
            result.append({
                "video_id": vid,
                "title": rec.get("title", ""),
                "channel": rec.get("channel", ""),
                "duration_seconds": rec.get("duration_seconds", 0),
                "source_url": rec.get("source_url", ""),
                "chunk_count": chunk_count,
                "has_cache": chunk_count is not None,
                "observation_count": obs_count,
            })
        result.sort(key=lambda r: r["title"].lower())
        self._send_json(result)

    def _handle_recording_detail(self, video_id):
        if not CACHE_SUBDIR:
            self._send_json({"error": "No inference cache directory found"}, 404)
            return
        cache_file = CACHE_SUBDIR / f"{video_id}.json"
        if not cache_file.is_file():
            self._send_json({"error": f"No cache for {video_id}"}, 404)
            return
        with open(cache_file, "r") as f:
            data = json.loads(f.read())
        self._send_json(data)

    def _handle_traces(self, video_id):
        if not TRACES_DIR.is_dir():
            self._send_json([])
            return
        traces = []
        prefix = f"{video_id}_chunk"
        for entry in sorted(TRACES_DIR.iterdir()):
            if entry.name.startswith(prefix) and entry.suffix == ".json":
                with open(entry, "r") as f:
                    traces.append(json.loads(f.read()))
        self._send_json(traces)

    def log_message(self, format, *args):
        # Quieter logging -- just method and path
        pass


def main():
    server = http.server.HTTPServer(("", PORT), ViewerHandler)
    print(f"Eval Viewer running at http://localhost:{PORT}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Cache dir:    {CACHE_SUBDIR or 'NOT FOUND'}")
    print(f"  Traces dir:   {TRACES_DIR}")
    print(f"  Audio dir:    {AUDIO_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
