# model/src/follower_eval/tap_tool.py
"""Human bar-tap gold-labeling tool for the real-audio follower eval (#133 S3).

A local web page (modeled on ``src/data_collection/review_candidates.py``) that,
per clip in the curated subset (``gold_subset.json``), serves the ACTUAL practice
WAV and lets a human tap each bar downbeat while listening. Each tap records the
audio ``currentTime`` against an auto-incrementing (and editable) bar number.
Saving POSTs the taps and writes
``data/evals/realaudio_bundles/<piece>/<vid>.gold.json`` -> the non-circular
ground truth the accuracy metric (``follower_eval.accuracy``) scores against.

WHY SERVE THE WAV, NOT THE YOUTUBE EMBED: the tap's ``audio_sec`` must live in
the SAME clock as the follower's ``MatchedNote.perf_time`` (the Transkun onset).
The served WAV IS the transcription's source, so ``currentTime`` == ``perf_time``
clock exactly; a YouTube re-encode could carry a different lead-in and shift
every tap. The WAV is streamed with HTTP Range support so the <audio> element can
seek (browsers require 206 responses to scrub).

RUNNING (from the PRIMARY checkout so data/ + the WAVs resolve):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.tap_tool --serve
  # then open http://localhost:8766
"""
from __future__ import annotations

import argparse
import datetime as dt
import http.server
import json
import socketserver
import sys
from dataclasses import dataclass
from pathlib import Path

SUBSET_JSON = Path(__file__).resolve().parent / "gold_subset.json"


@dataclass(frozen=True)
class GoldClip:
    """One subset clip resolved to its on-disk bundle + WAV + any prior taps."""
    piece: str
    video_id: str
    wav_path: Path
    title: str | None
    v1_confidence: float | None
    v1_coverage: float | None
    v1_span_frac: float | None
    existing_taps: list[dict]


class TapToolError(RuntimeError):
    """Raised when the subset, a bundle, or a WAV is missing -- loud, never a
    silent skip that would let the labeler tap a clip we can't score."""


def load_clips(subset_json: Path, bundles_root: Path,
               pieces: list[str] | None = None) -> list[GoldClip]:
    """Resolve every subset clip to its bundle (for the WAV path) + prior gold
    taps. Loud if a bundle or WAV is missing (the corpus must be built first)."""
    subset = json.loads(subset_json.read_text())
    clips: list[GoldClip] = []
    for c in subset["clips"]:
        piece, vid = c["piece"], c["video_id"]
        if pieces and piece not in pieces:
            continue
        bundle_path = bundles_root / piece / f"{vid}.json"
        if not bundle_path.exists():
            raise TapToolError(f"missing bundle {bundle_path} -- build the corpus first")
        bundle = json.loads(bundle_path.read_text())
        wav_path = Path(bundle["audio_path"])
        if not wav_path.exists():
            raise TapToolError(f"missing WAV {wav_path} for {piece}/{vid}")
        gold_path = bundles_root / piece / f"{vid}.gold.json"
        existing = []
        if gold_path.exists():
            existing = json.loads(gold_path.read_text()).get("bar_taps", [])
        clips.append(GoldClip(
            piece=piece, video_id=vid, wav_path=wav_path,
            title=c.get("title") or bundle.get("title"),
            v1_confidence=c.get("v1_confidence"), v1_coverage=c.get("v1_coverage"),
            v1_span_frac=c.get("v1_span_frac"), existing_taps=existing,
        ))
    if not clips:
        raise TapToolError(f"no clips resolved from {subset_json} (pieces={pieces})")
    return clips


def save_gold(bundles_root: Path, payload: dict) -> Path:
    """Write ``<piece>/<vid>.gold.json`` from a POSTed clip. Loud on empties --
    a save with zero taps is a mistake, not an empty answer key."""
    piece = payload["piece"]
    vid = payload["video_id"]
    taps = payload.get("bar_taps") or []
    if not taps:
        raise TapToolError(f"refusing to save {piece}/{vid}: zero taps")
    clean = [{"bar_number": int(t["bar_number"]), "audio_sec": round(float(t["audio_sec"]), 4)}
             for t in taps]
    clean.sort(key=lambda t: t["audio_sec"])
    out = {
        "piece": piece,
        "video_id": vid,
        "bar_taps": clean,
        "labeled_by": payload.get("labeled_by", "human"),
        "notes": payload.get("notes", ""),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    gold_path = bundles_root / piece / f"{vid}.gold.json"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(out, indent=1))
    return gold_path


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def generate_html(clips: list[GoldClip]) -> str:
    state = [
        {
            "piece": c.piece,
            "video_id": c.video_id,
            "title": c.title or c.video_id,
            "v1_confidence": c.v1_confidence,
            "v1_coverage": c.v1_coverage,
            "v1_span_frac": c.v1_span_frac,
            "existing_taps": c.existing_taps,
        }
        for c in clips
    ]
    n_labeled = sum(1 for c in clips if c.existing_taps)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>#133 Gold bar-tap labeler</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,monospace;
        background:#0a0a0a; color:#e0e0e0; padding:16px; }}
h1 {{ font-size:1.2rem; color:#fff; margin-bottom:2px; }}
.subtitle {{ color:#888; font-size:0.8rem; margin-bottom:12px; }}
.layout {{ display:flex; gap:16px; align-items:flex-start; }}
.clip-list {{ width:280px; flex-shrink:0; max-height:88vh; overflow-y:auto;
             border:1px solid #2a2a2a; border-radius:8px; padding:6px; }}
.clip-item {{ padding:6px 8px; cursor:pointer; border-radius:4px; font-size:0.78rem;
             border:1px solid transparent; }}
.clip-item:hover {{ background:#1a1a1a; }}
.clip-item.active {{ background:#22303a; border-color:#4a7a9a; }}
.clip-item .cp {{ color:#ddd; }}
.clip-item .meta {{ color:#777; font-size:0.7rem; }}
.clip-item.done .cp::before {{ content:"\\2713 "; color:#4ade80; }}
.panel {{ flex:1; border:1px solid #2a2a2a; border-radius:8px; padding:16px; }}
.panel h2 {{ font-size:1rem; color:#fff; margin-bottom:2px; }}
.panel .meta {{ color:#888; font-size:0.78rem; margin-bottom:12px; }}
audio {{ width:100%; margin-bottom:10px; }}
.controls {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:12px; }}
.tap-btn {{ padding:14px 28px; font-size:1.1rem; background:#1a4a1a; color:#4ade80;
           border:1px solid #4ade80; border-radius:8px; cursor:pointer; font-family:inherit; }}
.tap-btn:active {{ background:#2a6a2a; }}
.nextbar {{ display:flex; align-items:center; gap:6px; }}
.nextbar input {{ width:70px; padding:8px; font-size:1rem; background:#151515; color:#fff;
                 border:1px solid #444; border-radius:6px; font-family:inherit; text-align:center; }}
label {{ color:#aaa; font-size:0.8rem; }}
.rate button, .util button {{ padding:6px 10px; background:#1a1a1a; color:#aaa; border:1px solid #444;
                              border-radius:6px; cursor:pointer; font-family:inherit; font-size:0.8rem; }}
.rate button.active {{ background:#22303a; color:#fff; border-color:#4a7a9a; }}
.hint {{ color:#666; font-size:0.72rem; margin-bottom:10px; line-height:1.5; }}
.taps {{ max-height:38vh; overflow-y:auto; border:1px solid #2a2a2a; border-radius:6px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
th, td {{ padding:4px 8px; text-align:left; border-bottom:1px solid #222; }}
th {{ color:#888; position:sticky; top:0; background:#111; }}
td .del {{ color:#f87171; cursor:pointer; border:none; background:none; font-family:inherit; }}
.save-row {{ display:flex; align-items:center; gap:12px; margin-top:12px; }}
.save-btn {{ padding:8px 20px; background:#1a4a1a; color:#4ade80; border:1px solid #4ade80;
            border-radius:6px; cursor:pointer; font-family:inherit; }}
.clear-btn {{ padding:8px 14px; background:#2a1a1a; color:#f87171; border:1px solid #5a2d2d;
             border-radius:6px; cursor:pointer; font-family:inherit; }}
.status {{ color:#888; font-size:0.8rem; }}
kbd {{ background:#222; border:1px solid #444; border-radius:3px; padding:1px 5px; font-size:0.72rem; }}
</style>
</head>
<body>
<h1>#133 Real-audio follower &mdash; gold bar-tap labeler</h1>
<p class="subtitle">{len(clips)} clips &middot; {n_labeled} already labeled. Tap each bar's downbeat as you hear it.</p>

<div class="layout">
  <div class="clip-list" id="clip-list"></div>
  <div class="panel" id="panel"></div>
</div>

<script>
const CLIPS = {json.dumps(state)};
let cur = 0;                 // index into CLIPS
let taps = [];               // [{{bar_number, audio_sec}}]
let nextBar = 1;
let audioEl = null;

function pieceKey(c) {{ return c.piece + '/' + c.video_id; }}

function renderList() {{
  const el = document.getElementById('clip-list');
  el.innerHTML = CLIPS.map((c, i) => {{
    const done = (i === cur ? taps.length : (c.existing_taps || []).length) > 0;
    const conf = c.v1_confidence == null ? '?' : c.v1_confidence.toFixed(2);
    return `<div class="clip-item ${{i === cur ? 'active' : ''}} ${{done ? 'done' : ''}}"
                 onclick="selectClip(${{i}})">
      <div class="cp">${{c.piece}}</div>
      <div class="meta">${{c.video_id}} &middot; conf ${{conf}}</div>
    </div>`;
  }}).join('');
}}

function selectClip(i) {{
  // persist current taps into the in-memory CLIPS so switching doesn't lose them
  if (audioEl) audioEl.pause();
  CLIPS[cur].existing_taps = taps;
  cur = i;
  taps = (CLIPS[i].existing_taps || []).map(t => ({{bar_number: t.bar_number, audio_sec: t.audio_sec}}));
  nextBar = taps.length ? (taps[taps.length - 1].bar_number + 1) : 1;
  renderPanel();
  renderList();
}}

function renderPanel() {{
  const c = CLIPS[cur];
  const p = document.getElementById('panel');
  p.innerHTML = `
    <h2>${{c.piece}}</h2>
    <div class="meta">${{c.title || ''}} &middot; ${{c.video_id}} &middot;
      v1: conf ${{c.v1_confidence==null?'?':c.v1_confidence.toFixed(2)}},
      cov ${{c.v1_coverage==null?'?':c.v1_coverage.toFixed(2)}},
      span ${{c.v1_span_frac==null?'?':c.v1_span_frac.toFixed(2)}}</div>
    <audio id="audio" controls preload="auto"
           src="/audio/${{c.piece}}/${{c.video_id}}"></audio>
    <div class="hint">
      <kbd>Space</kbd> tap the current bar's downbeat &middot;
      <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> seek 2s &middot;
      <kbd>Backspace</kbd> undo last tap.<br>
      On a repeat/restart, set <b>Next bar</b> back to the bar you're about to hear, then keep tapping.
    </div>
    <div class="controls">
      <button class="tap-btn" onclick="tap()">TAP bar <span id="nb-lbl">${{nextBar}}</span></button>
      <div class="nextbar"><label>Next bar</label>
        <input id="nextbar-in" type="number" min="0" value="${{nextBar}}"
               onchange="setNextBar(this.value)"></div>
      <div class="rate">
        ${{[0.5,0.75,1].map(r => `<button data-r="${{r}}" onclick="setRate(${{r}})">${{r}}x</button>`).join('')}}
      </div>
    </div>
    <div class="taps"><table>
      <thead><tr><th>#</th><th>bar</th><th>audio_sec</th><th></th></tr></thead>
      <tbody id="taps-body"></tbody>
    </table></div>
    <div class="save-row">
      <button class="save-btn" onclick="saveClip()">Save this clip</button>
      <button class="clear-btn" onclick="clearTaps()">Clear taps</button>
      <span class="status" id="status"></span>
    </div>`;
  audioEl = document.getElementById('audio');
  setRate(1);
  renderTaps();
}}

function renderTaps() {{
  const body = document.getElementById('taps-body');
  body.innerHTML = taps.map((t, i) =>
    `<tr><td>${{i + 1}}</td><td>${{t.bar_number}}</td><td>${{t.audio_sec.toFixed(3)}}</td>
      <td><button class="del" onclick="delTap(${{i}})">del</button></td></tr>`).join('');
  document.getElementById('nb-lbl').textContent = nextBar;
  const inp = document.getElementById('nextbar-in');
  if (inp) inp.value = nextBar;
}}

function tap() {{
  if (!audioEl) return;
  taps.push({{bar_number: nextBar, audio_sec: audioEl.currentTime}});
  taps.sort((a, b) => a.audio_sec - b.audio_sec);
  nextBar += 1;
  renderTaps();
  renderList();
}}

function setNextBar(v) {{ nextBar = parseInt(v, 10); if (isNaN(nextBar)) nextBar = 1; renderTaps(); }}
function delTap(i) {{ taps.splice(i, 1); renderTaps(); renderList(); }}
function undo() {{
  if (!taps.length) return;
  const last = taps.pop();
  nextBar = last.bar_number;   // so you can re-tap that same bar
  renderTaps(); renderList();
}}
function clearTaps() {{ if (confirm('Clear all taps for this clip?')) {{ taps = []; nextBar = 1; renderTaps(); renderList(); }} }}
function setRate(r) {{
  if (audioEl) audioEl.playbackRate = r;
  document.querySelectorAll('.rate button').forEach(b =>
    b.classList.toggle('active', parseFloat(b.dataset.r) === r));
}}

function saveClip() {{
  const c = CLIPS[cur];
  const status = document.getElementById('status');
  if (!taps.length) {{ status.textContent = 'no taps to save'; return; }}
  status.textContent = 'saving...';
  fetch('/save-gold', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{piece: c.piece, video_id: c.video_id, bar_taps: taps}}),
  }}).then(r => r.json()).then(d => {{
    if (d.error) throw new Error(d.error);
    CLIPS[cur].existing_taps = taps.slice();
    status.textContent = 'saved ' + d.n + ' taps -> ' + d.path;
    renderList();
  }}).catch(e => {{ status.textContent = 'error: ' + e.message; }});
}}

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT') return;   // don't hijack the bar-number field
  if (e.code === 'Space') {{ e.preventDefault(); tap(); }}
  else if (e.code === 'Backspace') {{ e.preventDefault(); undo(); }}
  else if (e.code === 'ArrowLeft' && audioEl) {{ e.preventDefault(); audioEl.currentTime = Math.max(0, audioEl.currentTime - 2); }}
  else if (e.code === 'ArrowRight' && audioEl) {{ e.preventDefault(); audioEl.currentTime += 2; }}
}});

renderPanel();
renderList();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server (with Range support so the <audio> element can seek)
# ---------------------------------------------------------------------------


class TapHandler(http.server.BaseHTTPRequestHandler):
    clips: list[GoldClip]     # set by factory
    bundles_root: Path        # set by factory
    _wav_by_key: dict         # set by factory: "piece/vid" -> Path

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = generate_html(self.clips).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path.startswith("/audio/"):
            self._serve_wav(self.path[len("/audio/"):])
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_wav(self, key: str):
        wav = self._wav_by_key.get(key)
        if wav is None or not wav.exists():
            self.send_response(404)
            self.end_headers()
            return
        size = wav.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng and rng.startswith("bytes="):
            partial = True
            s, _, e = rng[len("bytes="):].partition("-")
            start = int(s) if s else 0
            end = int(e) if e else size - 1
            end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(wav, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return  # browser aborted the range (normal on seek) -- not an error
                remaining -= len(chunk)

    def do_POST(self):
        if self.path != "/save-gold":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            payload = json.loads(body)
            path = save_gold(self.bundles_root, payload)
            resp = {"n": len(payload.get("bar_taps", [])), "path": str(path.name)}
            code = 200
        except Exception as e:
            resp = {"error": str(e)}
            code = 400
        data = json.dumps(resp).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002 (match BaseHTTPRequestHandler)
        pass


def make_handler(clips: list[GoldClip], bundles_root: Path):
    wav_by_key = {f"{c.piece}/{c.video_id}": c.wav_path for c in clips}

    class Bound(TapHandler):
        pass

    Bound.clips = clips
    Bound.bundles_root = bundles_root
    Bound._wav_by_key = wav_by_key
    return Bound


def main() -> None:
    ap = argparse.ArgumentParser(description="Human bar-tap gold labeler (#133 S3)")
    ap.add_argument("--subset", type=Path, default=SUBSET_JSON)
    ap.add_argument("--bundles-root", type=Path, default=Path("data/evals/realaudio_bundles"))
    ap.add_argument("--pieces", nargs="+", default=None)
    ap.add_argument("--serve", action="store_true", help="start the local server")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    clips = load_clips(args.subset, args.bundles_root, pieces=args.pieces)
    print(f"Resolved {len(clips)} clips ({sum(1 for c in clips if c.existing_taps)} already labeled)")
    if not args.serve:
        print("Run with --serve to label. Clips:")
        for c in clips:
            print(f"  {c.piece}/{c.video_id}  ({len(c.existing_taps)} taps)  {c.wav_path}")
        return
    handler = make_handler(clips, args.bundles_root)
    # ThreadingTCPServer: a WAV Range stream must not block the /save-gold POST.
    with socketserver.ThreadingTCPServer(("", args.port), handler) as httpd:
        httpd.daemon_threads = True
        print(f"Gold labeler: http://localhost:{args.port}   (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    sys.exit(main())
