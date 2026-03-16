"""Local web UI for reviewing MAESTRO-to-ASAP match candidates.

Serves a single-page app with MIDI playback, match details,
and approve/reject buttons. Saves status directly to the CSV.

SECURITY NOTE: This tool is designed for LOCAL-ONLY use on localhost.
It serves files from the local filesystem and should never be exposed
to a network. All dynamic content is sanitized via textContent-based
escaping before DOM insertion.

Usage:
    cd model
    uv run python tools/review_matches.py \
        --csv data/reference_profiles/maestro_asap_matches.csv \
        --maestro-dir data/maestro_cache \
        --score-dir data/score_library

    Then open http://localhost:8765 in your browser.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# Globals set by CLI args
CSV_PATH: Path = Path()
MAESTRO_DIR: Path = Path()
SCORE_DIR: Path = Path()
MATCHES: list[dict] = []
FIELDNAMES: list[str] = []


def load_csv() -> None:
    global MATCHES, FIELDNAMES
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        FIELDNAMES = list(reader.fieldnames or [])
        MATCHES = list(reader)


def save_csv() -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in MATCHES:
            writer.writerow(row)


# The HTML is a static page. All dynamic data comes from /api/matches JSON
# and is rendered client-side using the esc() function which creates a
# temporary DOM text node (textContent) to safely escape any HTML entities.
# This prevents XSS even though we use DOM manipulation for the UI.
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Match Review</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #0a0a0a; color: #e0e0e0;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 24px;
  }
  .progress {
    width: 100%; max-width: 720px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 12px;
  }
  .progress-bar {
    flex: 1; height: 6px; background: #222; border-radius: 3px; overflow: hidden;
  }
  .progress-fill { height: 100%; background: #4a9; border-radius: 3px; transition: width 0.3s; }
  .progress-text { font-size: 13px; color: #888; min-width: 120px; text-align: right; }
  .stats {
    display: flex; gap: 16px; margin-bottom: 20px; font-size: 13px;
  }
  .stat { padding: 4px 10px; border-radius: 4px; }
  .stat-approved { background: #1a3a2a; color: #4a9; }
  .stat-rejected { background: #3a1a1a; color: #c55; }
  .stat-pending { background: #2a2a1a; color: #aa8; }
  .card {
    width: 100%; max-width: 720px; background: #141414;
    border: 1px solid #2a2a2a; border-radius: 12px;
    padding: 28px; margin-bottom: 20px;
  }
  .confidence {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 13px; font-weight: 600; margin-bottom: 16px;
  }
  .conf-high { background: #1a3a2a; color: #4a9; }
  .conf-mid { background: #2a2a1a; color: #aa8; }
  .conf-low { background: #3a1a1a; color: #c55; }
  .multi-piece {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 13px; font-weight: 600; margin-left: 8px;
    background: #3a2a1a; color: #c84;
  }
  .match-info { margin-bottom: 20px; }
  .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-bottom: 4px; }
  .value { font-size: 18px; font-weight: 500; margin-bottom: 16px; }
  .value-small { font-size: 14px; color: #999; margin-bottom: 12px; }
  .midi-filename { font-size: 12px; color: #555; word-break: break-all; margin-bottom: 16px; }
  .player {
    display: flex; align-items: center; gap: 12px; margin-bottom: 8px;
    padding: 12px; background: #1a1a1a; border-radius: 8px;
  }
  .play-btn {
    width: 44px; height: 44px; border-radius: 50%;
    background: #2a2a2a; border: none; color: #e0e0e0;
    font-size: 18px; cursor: pointer; display: flex;
    align-items: center; justify-content: center;
    transition: background 0.15s;
  }
  .play-btn:hover { background: #3a3a3a; }
  .play-btn.playing { background: #4a9; color: #000; }
  .time-display { font-size: 13px; color: #888; font-variant-numeric: tabular-nums; }
  .actions {
    display: flex; gap: 12px; margin-top: 20px;
  }
  .btn {
    flex: 1; padding: 14px 24px; border: none; border-radius: 8px;
    font-size: 15px; font-weight: 600; cursor: pointer;
    transition: all 0.15s; text-transform: uppercase; letter-spacing: 1px;
  }
  .btn-approve { background: #1a3a2a; color: #4a9; }
  .btn-approve:hover { background: #2a5a3a; }
  .btn-reject { background: #3a1a1a; color: #c55; }
  .btn-reject:hover { background: #5a2a2a; }
  .btn-skip { background: #1a1a2a; color: #88a; }
  .btn-skip:hover { background: #2a2a3a; }
  .shortcuts {
    font-size: 12px; color: #555; text-align: center; margin-top: 8px;
  }
  .filter-bar {
    display: flex; gap: 8px; margin-bottom: 16px; width: 100%; max-width: 720px;
  }
  .filter-btn {
    padding: 6px 14px; border: 1px solid #333; border-radius: 6px;
    background: transparent; color: #888; font-size: 13px; cursor: pointer;
  }
  .filter-btn.active { border-color: #4a9; color: #4a9; }
  .done-msg {
    font-size: 20px; color: #4a9; text-align: center; padding: 40px;
  }
</style>
</head>
<body>

<div class="progress">
  <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
  <div class="progress-text" id="progressText">0 / 0</div>
</div>

<div class="stats">
  <span class="stat stat-approved" id="statApproved">0 approved</span>
  <span class="stat stat-rejected" id="statRejected">0 rejected</span>
  <span class="stat stat-pending" id="statPending">0 pending</span>
</div>

<div class="filter-bar" id="filterBar"></div>

<div class="card" id="card">
  <div id="content">Loading...</div>
</div>

<div class="shortcuts">
  Keyboard: <strong>A</strong> = Approve &nbsp; <strong>R</strong> = Reject &nbsp;
  <strong>S</strong> = Skip &nbsp; <strong>1</strong> = Play MAESTRO &nbsp; <strong>2</strong> = Play Score &nbsp;
  <strong>Space</strong> = Play/Stop &nbsp; <strong>Arrow keys</strong> = Prev/Next
</div>

<script src="https://cdn.jsdelivr.net/npm/tone@14/build/Tone.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tonejs/midi@2/build/Midi.js"></script>
<script>
// -- Safe DOM helpers (no innerHTML with untrusted data) --
function createEl(tag, attrs, children) {
  const el = document.createElement(tag);
  if (attrs) {
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'className') el.className = v;
      else if (k === 'textContent') el.textContent = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
      else el.setAttribute(k, v);
    });
  }
  if (children) {
    (Array.isArray(children) ? children : [children]).forEach(c => {
      if (typeof c === 'string') el.appendChild(document.createTextNode(c));
      else if (c) el.appendChild(c);
    });
  }
  return el;
}

let matches = [];
let filtered = [];
let currentIdx = 0;
let currentFilter = 'pending';
let synth = null;
let activeSource = null; // 'maestro' or 'score'
let playbackTimer = null;
let playbackStart = 0;

async function loadMatches() {
  const resp = await fetch('/api/matches');
  matches = await resp.json();
  buildFilterBar();
  applyFilter();
  render();
}

function buildFilterBar() {
  const bar = document.getElementById('filterBar');
  bar.replaceChildren();
  ['pending', 'all', 'approved', 'rejected'].forEach(f => {
    const btn = createEl('button', {
      className: 'filter-btn' + (f === currentFilter ? ' active' : ''),
      textContent: f.charAt(0).toUpperCase() + f.slice(1),
      onClick: () => setFilter(f),
      'data-filter': f,
    });
    bar.appendChild(btn);
  });
}

function applyFilter() {
  if (currentFilter === 'all') {
    filtered = matches.slice();
  } else if (currentFilter === 'pending') {
    filtered = matches.filter(m => !m.status || m.status.trim() === '');
  } else {
    filtered = matches.filter(m => m.status && m.status.trim().toLowerCase() === currentFilter);
  }
  currentIdx = 0;
}

function setFilter(f) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.filter === f);
  });
  applyFilter();
  stopPlayback();
  render();
}

function updateStats() {
  const approved = matches.filter(m => m.status && m.status.trim().toLowerCase() === 'approved').length;
  const rejected = matches.filter(m => m.status && m.status.trim().toLowerCase() === 'rejected').length;
  const pending = matches.length - approved - rejected;
  document.getElementById('statApproved').textContent = approved + ' approved';
  document.getElementById('statRejected').textContent = rejected + ' rejected';
  document.getElementById('statPending').textContent = pending + ' pending';

  const reviewed = approved + rejected;
  const pct = matches.length > 0 ? (reviewed / matches.length * 100) : 0;
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent =
    reviewed + ' / ' + matches.length + ' reviewed';
}

function render() {
  updateStats();
  const container = document.getElementById('content');
  container.replaceChildren();

  if (filtered.length === 0) {
    container.appendChild(createEl('div', {className: 'done-msg', textContent: 'No matches to show for this filter.'}));
    return;
  }

  if (currentIdx >= filtered.length) currentIdx = filtered.length - 1;
  const m = filtered[currentIdx];
  const conf = parseFloat(m.confidence);
  const confClass = conf >= 0.7 ? 'conf-high' : conf >= 0.4 ? 'conf-mid' : 'conf-low';

  // Confidence badge
  const confBadge = createEl('span', {
    className: 'confidence ' + confClass,
    textContent: Math.round(conf * 100) + '% confidence',
  });
  container.appendChild(confBadge);

  // Multi-piece badge
  if (m.multi_piece === 'True') {
    container.appendChild(createEl('span', {className: 'multi-piece', textContent: 'MULTI-PIECE'}));
  }

  // Status badge
  if (m.status && m.status.trim()) {
    const st = m.status.trim().toLowerCase();
    container.appendChild(createEl('span', {
      className: 'confidence ' + (st === 'approved' ? 'conf-high' : 'conf-low'),
      textContent: m.status.trim().toUpperCase(),
      style: 'margin-left:8px',
    }));
  }

  // Match info
  const info = createEl('div', {className: 'match-info'});

  info.appendChild(createEl('div', {className: 'label', textContent: 'MAESTRO Recording'}));
  info.appendChild(createEl('div', {className: 'value', textContent: m.maestro_composer + ' - ' + m.maestro_title}));

  info.appendChild(createEl('div', {className: 'label', textContent: 'Matched ASAP Piece'}));
  info.appendChild(createEl('div', {className: 'value', textContent: m.asap_title}));
  info.appendChild(createEl('div', {className: 'value-small', textContent: m.asap_piece_id}));

  const durMin = (parseFloat(m.duration_s) / 60).toFixed(1);
  info.appendChild(createEl('div', {className: 'midi-filename', textContent: m.midi_filename + ' (' + durMin + ' min)'}));
  container.appendChild(info);

  // Dual players
  const playerRow = createEl('div', {style: 'display:flex;gap:12px;margin-bottom:8px'});

  const mPlayer = createEl('div', {className: 'player', style: 'flex:1'}, [
    createEl('button', {className: 'play-btn', id: 'playMaestro', textContent: '\u25B6', onClick: () => playMidi('maestro')}),
    createEl('span', {style: 'flex:1'}),
    createEl('span', {className: 'time-display', textContent: 'MAESTRO', style: 'color:#6af'}),
    createEl('span', {className: 'time-display', id: 'timeMaestro', textContent: '0:00'}),
  ]);
  playerRow.appendChild(mPlayer);

  const sPlayer = createEl('div', {className: 'player', style: 'flex:1'}, [
    createEl('button', {className: 'play-btn', id: 'playScore', textContent: '\u25B6', onClick: () => playMidi('score')}),
    createEl('span', {style: 'flex:1'}),
    createEl('span', {className: 'time-display', textContent: 'SCORE', style: 'color:#fa6'}),
    createEl('span', {className: 'time-display', id: 'timeScore', textContent: '0:00'}),
  ]);
  playerRow.appendChild(sPlayer);
  container.appendChild(playerRow);

  // Action buttons
  const actions = createEl('div', {className: 'actions'}, [
    createEl('button', {className: 'btn btn-approve', textContent: 'Approve (A)', onClick: () => setStatus('approved')}),
    createEl('button', {className: 'btn btn-skip', textContent: 'Skip (S)', onClick: () => advance(1)}),
    createEl('button', {className: 'btn btn-reject', textContent: 'Reject (R)', onClick: () => setStatus('rejected')}),
  ]);
  container.appendChild(actions);

  // Nav
  const nav = createEl('div', {style: 'display:flex;justify-content:space-between;margin-top:12px'}, [
    createEl('button', {className: 'btn btn-skip', style: 'flex:none;padding:8px 16px;font-size:12px', textContent: '\u2190 Prev', onClick: () => advance(-1)}),
    createEl('span', {style: 'color:#555;font-size:13px;line-height:36px', textContent: (currentIdx + 1) + ' / ' + filtered.length}),
    createEl('button', {className: 'btn btn-skip', style: 'flex:none;padding:8px 16px;font-size:12px', textContent: 'Next \u2192', onClick: () => advance(1)}),
  ]);
  container.appendChild(nav);
}

async function setStatus(status) {
  if (filtered.length === 0) return;
  const m = filtered[currentIdx];
  const masterIdx = matches.findIndex(x => x.midi_filename === m.midi_filename);
  if (masterIdx >= 0) {
    matches[masterIdx].status = status;
    m.status = status;
  }
  await fetch('/api/status', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({midi_filename: m.midi_filename, status: status})
  });
  stopPlayback();
  if (currentFilter === 'pending') {
    filtered.splice(currentIdx, 1);
    if (currentIdx >= filtered.length) currentIdx = Math.max(0, filtered.length - 1);
  } else {
    advance(1);
  }
  render();
}

function advance(delta) {
  stopPlayback();
  currentIdx = Math.max(0, Math.min(filtered.length - 1, currentIdx + delta));
  render();
}

function ensureSynth() {
  if (!synth) {
    synth = new Tone.PolySynth(Tone.Synth, {
      maxPolyphony: 64,
      voice: Tone.Synth,
      options: {
        envelope: { attack: 0.01, decay: 0.2, sustain: 0.3, release: 0.6 },
        oscillator: { type: 'triangle8' },
      }
    }).toDestination();
    synth.volume.value = -6;
  }
  return synth;
}

async function playMidi(source) {
  // If already playing this source, stop it
  if (activeSource === source) { stopPlayback(); return; }
  // If playing something else, stop first
  if (activeSource) stopPlayback();

  if (filtered.length === 0) return;
  const m = filtered[currentIdx];

  const btnId = source === 'maestro' ? 'playMaestro' : 'playScore';
  const timeId = source === 'maestro' ? 'timeMaestro' : 'timeScore';
  const btn = document.getElementById(btnId);
  if (btn) btn.textContent = '...';

  try {
    await Tone.start();
    const s = ensureSynth();

    let notes = [];
    if (source === 'maestro') {
      // Load MIDI file via @tonejs/midi
      const resp = await fetch('/midi/' + encodeURIComponent(m.midi_filename));
      if (!resp.ok) throw new Error('MIDI fetch failed: ' + resp.status);
      const buf = await resp.arrayBuffer();
      const midi = new Midi(buf);
      midi.tracks.forEach(track => {
        track.notes.forEach(note => {
          notes.push({ time: note.time, midi: note.midi, dur: note.duration, vel: note.velocity });
        });
      });
    } else {
      // Load score JSON and extract notes
      const resp = await fetch('/score/' + encodeURIComponent(m.asap_piece_id + '.json'));
      if (!resp.ok) throw new Error('Score fetch failed: ' + resp.status);
      const score = await resp.json();
      (score.bars || []).forEach(bar => {
        (bar.notes || []).forEach(n => {
          notes.push({
            time: n.onset_seconds,
            midi: n.pitch,
            dur: n.duration_seconds,
            vel: (n.velocity || 80) / 127,
          });
        });
      });
    }

    // Sort by time
    notes.sort((a, b) => a.time - b.time);

    // Schedule first 30 seconds using direct triggerAttackRelease
    // This avoids the Transport.schedule bug
    const now = Tone.now() + 0.15;
    const maxTime = 30;
    for (const n of notes) {
      if (n.time > maxTime) break;
      const noteName = Tone.Frequency(n.midi, 'midi').toNote();
      const dur = Math.max(0.05, Math.min(n.dur, 2.0));
      const vel = Math.max(0.05, Math.min(n.vel, 1.0));
      s.triggerAttackRelease(noteName, dur, now + n.time, vel);
    }

    activeSource = source;
    playbackStart = Tone.now();
    if (btn) { btn.textContent = '\u25A0'; btn.classList.add('playing'); }

    // Update time display
    playbackTimer = setInterval(() => {
      if (!activeSource) { clearInterval(playbackTimer); return; }
      const elapsed = Tone.now() - playbackStart;
      if (elapsed > maxTime) { stopPlayback(); return; }
      const mins = Math.floor(elapsed / 60);
      const secs = Math.floor(elapsed % 60);
      const td = document.getElementById(timeId);
      if (td) td.textContent = mins + ':' + String(secs).padStart(2, '0');
    }, 250);

  } catch (e) {
    console.error('Playback error:', e);
    if (btn) btn.textContent = '\u25B6';
  }
}

function stopPlayback() {
  if (!activeSource) return;
  const btnId = activeSource === 'maestro' ? 'playMaestro' : 'playScore';
  const timeId = activeSource === 'maestro' ? 'timeMaestro' : 'timeScore';
  activeSource = null;
  if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; }
  if (synth) synth.releaseAll();
  const btn = document.getElementById(btnId);
  if (btn) { btn.textContent = '\u25B6'; btn.classList.remove('playing'); }
  const td = document.getElementById(timeId);
  if (td) td.textContent = '0:00';
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'a' || e.key === 'A') setStatus('approved');
  else if (e.key === 'r' || e.key === 'R') setStatus('rejected');
  else if (e.key === 's' || e.key === 'S') advance(1);
  else if (e.key === '1') playMidi('maestro');
  else if (e.key === '2') playMidi('score');
  else if (e.key === ' ') { e.preventDefault(); if (activeSource) stopPlayback(); else playMidi('maestro'); }
  else if (e.key === 'ArrowLeft') advance(-1);
  else if (e.key === 'ArrowRight') advance(1);
});

loadMatches();
</script>
</body>
</html>
"""


class ReviewHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed.path == "/api/matches":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(MATCHES).encode("utf-8"))

        elif parsed.path.startswith("/midi/"):
            # Serve MIDI file from maestro_dir
            midi_rel = unquote(parsed.path[6:])  # strip "/midi/"
            # Prevent path traversal
            midi_path = (MAESTRO_DIR / midi_rel).resolve()
            if not str(midi_path).startswith(str(MAESTRO_DIR)):
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"Forbidden")
                return
            if midi_path.exists() and midi_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "audio/midi")
                self.send_header("Content-Length", str(midi_path.stat().st_size))
                self.end_headers()
                with open(midi_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"MIDI file not found")

        elif parsed.path.startswith("/score/"):
            # Serve score JSON from score_dir
            score_name = unquote(parsed.path[7:])  # strip "/score/"
            score_path = (SCORE_DIR / score_name).resolve()
            if not str(score_path).startswith(str(SCORE_DIR)):
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"Forbidden")
                return
            if score_path.exists() and score_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with open(score_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Score file not found")

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/status":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            midi_filename = body.get("midi_filename", "")
            status = body.get("status", "")

            # Validate status value
            if status not in ("approved", "rejected"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "invalid status"}).encode("utf-8"))
                return

            updated = False
            for row in MATCHES:
                if row["midi_filename"] == midi_filename:
                    row["status"] = status
                    updated = True
                    break

            if updated:
                save_csv()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": updated}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Review MAESTRO-to-ASAP matches in a browser UI")
    parser.add_argument("--csv", type=Path, required=True, help="Path to maestro_asap_matches.csv")
    parser.add_argument("--maestro-dir", type=Path, required=True, help="Root directory of MAESTRO MIDI cache")
    parser.add_argument("--score-dir", type=Path, required=True, help="Directory containing score JSON files")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765)")
    args = parser.parse_args()

    global CSV_PATH, MAESTRO_DIR, SCORE_DIR
    CSV_PATH = args.csv.resolve()
    MAESTRO_DIR = args.maestro_dir.resolve()
    SCORE_DIR = args.score_dir.resolve()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not MAESTRO_DIR.exists():
        raise FileNotFoundError(f"MAESTRO directory not found: {MAESTRO_DIR}")
    if not SCORE_DIR.exists():
        raise FileNotFoundError(f"Score directory not found: {SCORE_DIR}")

    load_csv()
    pending = sum(1 for m in MATCHES if not m.get("status", "").strip())
    reviewed = len(MATCHES) - pending
    print(f"Loaded {len(MATCHES)} matches ({reviewed} reviewed, {pending} pending)")
    print(f"MAESTRO dir: {MAESTRO_DIR}")
    print(f"Score dir: {SCORE_DIR}")
    print(f"")
    print(f"  Open http://localhost:{args.port} in your browser")
    print(f"  Press Ctrl+C to stop")
    print()

    server = HTTPServer(("localhost", args.port), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
