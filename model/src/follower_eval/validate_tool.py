# model/src/follower_eval/validate_tool.py
# ruff: noqa: E501
"""Light-touch human VALIDATION tool for the amateur clips (issue #133, Track B).

The tap tool asked the impossible (name the bar by ear, with no score). This
flips the labor: the human WATCHES and flags, never reads a score or a bar
number. For a selected clip it draws two note-strips on one shared score-time
axis, with a playhead driven by the follower's decoded position:

  * BOTTOM (reference)  -- the score's notes at their score positions.
  * TOP (what you played, as the follower placed it) -- each transcribed note
    mapped to the score position the follower thinks it belongs to
    (``decode_at`` of its onset).

When the follower is right, each played note sits directly above its matching
score note (same pitch under the playhead). When it's wrong, played notes pile
onto the wrong score location and the pitches under the playhead clash -- visible
AND audible against the real audio. The human holds SPACE across wrong spans and
picks one per-clip verdict. Output: fraction-of-playback-correct + a verdict that
adjudicates the low-confidence proxy clips (genuinely-hard vs follower-failure).

Per-clip follower data is computed lazily on selection (one ``follow_hmm`` run,
cached) and the WAV is streamed with HTTP Range so the <audio> element can seek.

RUNNING (from the PRIMARY checkout so data/ + the WAVs resolve):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.validate_tool --serve
  # then open http://localhost:8767
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.server
import json
import socketserver
import statistics
import sys
from pathlib import Path

from follower_bench.hmm import TUNED_HMM_PARAMS, follow_hmm
from follower_eval.accuracy import decode_at
from follower_eval.realaudio import (
    SCORE_FILENAME_BY_PIECE,
    load_bundle_notes,
    load_score,
)

SUBSET_JSON = Path(__file__).resolve().parent / "gold_subset.json"
VALID_VERDICTS = {"tracked", "recovered", "wrong", "junk"}


class ValidateToolError(RuntimeError):
    """Loud failure when a bundle, WAV, or score is missing -- never a silent
    skip that would let the human validate a clip we can't render."""


def load_piece_id_map(path: Path) -> dict[str, str]:
    """``{"<piece>/<vid>": score_id}`` from a ``piece_id`` run, ABSTAINs omitted.

    The corpus folder labels are unreliable (a clip filed `fantaisie_impromptu`
    is Chopin Op.25/5), so the validator must render the score the piece-ID
    stage says is actually being played -- otherwise the human is shown a
    mismatch and blames the follower for correctly refusing to track a piece it
    was never given.
    """
    if not path.exists():
        raise ValidateToolError(
            f"no piece-ID map at {path}. The corpus labels are known-wrong, so the "
            f"validator refuses to guess. Run:\n"
            f"  python -m follower_eval.piece_id --clips <piece/vid ...> --out {path}\n"
            f"or pass --trust-labels to validate against the folder label anyway."
        )
    out: dict[str, str] = {}
    for r in json.loads(path.read_text()):
        if r["decision"] != "ABSTAIN":
            out[f"{r['piece_folder']}/{r['video_id']}"] = r["decision"]
    return out


def resolve_score_id(piece: str, vid: str, id_map: dict[str, str]) -> tuple[str, str]:
    """``(score_id, source)`` for a clip. ``source`` is ``"piece_id"`` when the
    identification stage named the score, else ``"label"`` -- the fallback for
    clips it abstained on, surfaced in the UI so the human knows the score on
    screen is unverified rather than assuming a follower failure."""
    sid = id_map.get(f"{piece}/{vid}")
    if sid is not None:
        return sid, "piece_id"
    if piece not in SCORE_FILENAME_BY_PIECE:
        raise ValidateToolError(f"no score for piece {piece} and piece-ID abstained")
    return SCORE_FILENAME_BY_PIECE[piece].removesuffix(".json"), "label"


def list_clips(
    subset_json: Path,
    bundles_root: Path,
    use_all: bool,
    pieces: list[str] | None,
    id_map: dict[str, str] | None = None,
) -> list[dict]:
    """Resolve the clips to validate -> [{piece, video_id, wav_path, title,
    v1_confidence, existing, score_id, score_source}]. Default source is the
    committed gold subset (spans confidence); ``use_all`` enumerates every bundle
    instead. ``id_map`` comes from ``load_piece_id_map`` and decides which score
    each clip is validated against."""
    id_map = id_map or {}
    entries: list[dict] = []
    if use_all:
        for piece_dir in sorted(p for p in bundles_root.iterdir() if p.is_dir()):
            if piece_dir.name not in SCORE_FILENAME_BY_PIECE:
                continue
            for b in sorted(piece_dir.glob("*.json")):
                if (
                    b.name.endswith(".meta.json")
                    or b.name in ("_index.json",)
                    or b.name.endswith(".gold.json")
                    or b.name.endswith(".validate.json")
                ):
                    continue
                entries.append(
                    {"piece": piece_dir.name, "video_id": b.stem, "v1_confidence": None}
                )
    else:
        for c in json.loads(subset_json.read_text())["clips"]:
            entries.append(
                {
                    "piece": c["piece"],
                    "video_id": c["video_id"],
                    "v1_confidence": c.get("v1_confidence"),
                }
            )

    out: list[dict] = []
    for e in entries:
        piece, vid = e["piece"], e["video_id"]
        if pieces and piece not in pieces:
            continue
        bundle_path = bundles_root / piece / f"{vid}.json"
        if not bundle_path.exists():
            raise ValidateToolError(f"missing bundle {bundle_path}")
        bundle = json.loads(bundle_path.read_text())
        wav = Path(bundle["audio_path"])
        if not wav.exists():
            raise ValidateToolError(f"missing WAV {wav} for {piece}/{vid}")
        vpath = bundles_root / piece / f"{vid}.validate.json"
        # Re-selecting a clip must restore what was already saved for it, not blank
        # the UI: a silent reset is how marked spans get overwritten with [].
        saved = json.loads(vpath.read_text()) if vpath.exists() else {}
        score_id, score_source = resolve_score_id(piece, vid, id_map)
        label_sid = SCORE_FILENAME_BY_PIECE.get(piece, "").removesuffix(".json")
        out.append(
            {
                "relabeled": score_source == "piece_id" and score_id != label_sid,
                "piece": piece,
                "video_id": vid,
                "wav_path": wav,
                "title": bundle.get("title"),
                "v1_confidence": e["v1_confidence"],
                "existing": vpath.exists(),
                "saved_spans": saved.get("wrong_spans", []),
                "saved_verdict": saved.get("verdict"),
                "score_id": score_id,
                "score_source": score_source,
            }
        )
    if not out:
        raise ValidateToolError(f"no clips resolved (all={use_all}, pieces={pieces})")
    # low-confidence first: those are the clips the validation most needs to adjudicate
    out.sort(
        key=lambda c: (
            c["v1_confidence"] is None,
            c["v1_confidence"] if c["v1_confidence"] is not None else 1.0,
        )
    )
    return out


def build_clip_view(
    piece: str,
    bundle_path: Path,
    scores_root: Path,
    score_id: str,
    score_source: str = "piece_id",
) -> dict:
    """Run the follower once and precompute everything the canvas needs: the
    score notes, the played notes mapped to score position by the follower, the
    decoded trajectory (for the playhead), and the median confidence.

    ``score_id`` is the score to follow against -- resolved by ``resolve_score_id``
    from the piece-ID stage, NOT from the (unreliable) folder label.

    Raises:
        ValidateToolError: the score file is missing, or loaders fail loudly.
    """
    score_path = scores_root / f"{score_id}.json"
    if not score_path.exists():
        raise ValidateToolError(
            f"missing score {score_path} for {piece}/{bundle_path.stem}"
        )
    score_notes, bar_boundaries, score_span = load_score(score_path)
    perf = load_bundle_notes(bundle_path)
    est = follow_hmm(perf, score_notes, TUNED_HMM_PARAMS, bar_boundaries=bar_boundaries)

    ms = sorted(est.matches, key=lambda m: m.perf_time)
    pt = [m.perf_time for m in ms]
    sp = [m.score_position for m in ms]
    confs = [m.confidence for m in ms if m.confidence is not None]

    # each played note -> the score position the follower places it at
    played = [
        {
            "sx": round(decode_at(pt, sp, n.onset) or 0.0, 3),
            "t": round(n.onset, 3),
            "pitch": n.pitch,
        }
        for n in perf
    ]
    score = [{"sx": round(n.position, 3), "pitch": n.pitch} for n in score_notes]
    return {
        "piece": piece,
        "score_id": score_id,
        "score_source": score_source,
        "score_span_sec": round(score_span, 2),
        "played": played,
        "score": score,
        # decoded trajectory (audio_sec -> score_sec) for the JS playhead
        "traj_t": [round(x, 3) for x in pt],
        "traj_s": [round(x, 3) for x in sp],
        "median_confidence": round(statistics.median(confs), 3) if confs else None,
        "transpose_semitones": est.transpose_semitones,
    }


def _view_cache_path(bundles_root: Path, piece: str, vid: str, score_id: str) -> Path:
    # score_id is part of the key: a clip re-labeled by piece-ID must not be
    # served the view that was computed against its old (wrong) label score.
    return bundles_root / "_view_cache" / f"{piece}__{vid}__{score_id}.view.json"


def get_clip_view(
    piece: str,
    vid: str,
    bundles_root: Path,
    scores_root: Path,
    score_id: str,
    score_source: str = "piece_id",
    force: bool = False,
) -> dict:
    """build_clip_view with a disk cache. ``follow_hmm`` is O(perf x score) and
    the big amateur clips take minutes -- caching the precomputed view to
    ``_view_cache/`` makes the labeler load instantly. Cache is keyed by clip AND
    score (follower params are fixed); pass ``force`` after a follower change."""
    cache = _view_cache_path(bundles_root, piece, vid, score_id)
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    view = build_clip_view(
        piece, bundles_root / piece / f"{vid}.json", scores_root, score_id, score_source
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(view))
    return view


def save_validation(bundles_root: Path, payload: dict) -> Path:
    """Write ``<piece>/<vid>.validate.json`` from a POSTed validation."""
    piece, vid = payload["piece"], payload["video_id"]
    if (
        Path(piece).name != piece
        or Path(vid).name != vid
        or piece in {".", ".."}
        or vid in {".", ".."}
    ):
        raise ValidateToolError(f"unsafe clip identifier {piece}/{vid}")
    verdict = payload.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValidateToolError(
            f"refusing to save {piece}/{vid}: verdict must be one of {sorted(VALID_VERDICTS)}"
        )
    provenance = ("score_id", "score_source", "follower_confidence")
    missing = [field for field in provenance if field not in payload]
    if missing:
        raise ValidateToolError(
            f"refusing to save {piece}/{vid}: missing score provenance {missing}; "
            "reload the validator so confidence is computed against the resolved score"
        )
    spans = [
        [round(float(a), 3), round(float(b), 3)]
        for a, b in payload.get("wrong_spans", [])
    ]
    out = {
        "piece": piece,
        "video_id": vid,
        "verdict": verdict,  # tracked | recovered | wrong | junk
        "score_id": payload["score_id"],
        "score_source": payload["score_source"],
        "follower_confidence": payload["follower_confidence"],
        "wrong_spans": spans,
        "audio_duration_sec": payload.get("audio_duration_sec"),
        "fraction_wrong": payload.get("fraction_wrong"),
        "validated_by": payload.get("validated_by", "human"),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    p = bundles_root / piece / f"{vid}.validate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return p


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def generate_html(clips: list[dict]) -> str:
    state = [
        {
            "piece": c["piece"],
            "video_id": c["video_id"],
            "title": c["title"] or c["video_id"],
            "v1_confidence": c["v1_confidence"],
            "existing": c["existing"],
            "saved_spans": c["saved_spans"],
            "saved_verdict": c["saved_verdict"],
            "score_id": c["score_id"],
            "score_source": c["score_source"],
            "relabeled": c["relabeled"],
        }
        for c in clips
    ]
    n_done = sum(1 for c in clips if c["existing"])
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>#133 follower validator</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",monospace; background:#0a0a0a; color:#e0e0e0; padding:14px; }}
h1 {{ font-size:1.1rem; color:#fff; }}
.subtitle {{ color:#888; font-size:0.78rem; margin-bottom:10px; }}
.layout {{ display:flex; gap:14px; align-items:flex-start; }}
.clip-list {{ width:250px; flex-shrink:0; max-height:90vh; overflow-y:auto; border:1px solid #2a2a2a; border-radius:8px; padding:6px; }}
.clip-item {{ padding:6px 8px; cursor:pointer; border-radius:4px; font-size:0.76rem; border:1px solid transparent; }}
.clip-item:hover {{ background:#1a1a1a; }}
.clip-item.active {{ background:#22303a; border-color:#4a7a9a; }}
.clip-item .meta {{ color:#777; font-size:0.68rem; }}
.clip-item.done .cp::before {{ content:"\\2713 "; color:#4ade80; }}
.panel {{ flex:1; border:1px solid #2a2a2a; border-radius:8px; padding:14px; min-width:0; }}
.panel h2 {{ font-size:0.95rem; color:#fff; }}
.panel .meta {{ color:#888; font-size:0.76rem; margin-bottom:8px; }}
canvas {{ width:100%; height:320px; background:#0d0d0d; border:1px solid #222; border-radius:6px; display:block; }}
audio {{ width:100%; margin:8px 0; }}
.legend {{ font-size:0.72rem; color:#999; margin:4px 0 8px; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:middle; margin:0 3px 0 10px; }}
.hint {{ color:#666; font-size:0.72rem; margin-bottom:8px; }}
kbd {{ background:#222; border:1px solid #444; border-radius:3px; padding:1px 5px; }}
.verdicts {{ display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }}
.verdict-btn {{ padding:7px 12px; background:#151515; color:#bbb; border:1px solid #444; border-radius:6px; cursor:pointer; font-family:inherit; font-size:0.8rem; }}
.verdict-btn.sel {{ background:#22303a; color:#fff; border-color:#4a7a9a; }}
.save-row {{ display:flex; align-items:center; gap:12px; margin-top:6px; }}
.save-btn {{ padding:8px 18px; background:#1a4a1a; color:#4ade80; border:1px solid #4ade80; border-radius:6px; cursor:pointer; font-family:inherit; }}
.status {{ color:#888; font-size:0.8rem; }}
.wrongbar {{ height:14px; background:#141414; border:1px solid #222; border-radius:3px; position:relative; margin:6px 0; overflow:hidden; }}
.wrongbar .seg {{ position:absolute; top:0; bottom:0; background:#7a2d2d; }}
.wrongbar .ph {{ position:absolute; top:0; bottom:0; width:2px; background:#4ade80; }}
</style></head><body>
<h1>#133 follower validator &mdash; watch &amp; flag</h1>
<p class="subtitle">{len(clips)} clips &middot; {n_done} validated. Low-confidence first. No scores to read &mdash; just watch the two rows.</p>
<div class="layout">
  <div class="clip-list" id="clip-list"></div>
  <div class="panel" id="panel"><p class="meta">Select a clip on the left.</p></div>
</div>
<script>
const CLIPS = {json.dumps(state)};
let cur = -1, view = null, audioEl = null, canvas = null, ctx = null, raf = null;
let wrongSpans = [], spanOpen = null, verdict = null;
const PX_PER_SCORE_SEC = 40, PITCH_MIN = 21, PITCH_MAX = 108;

// Which score is on screen, and can we trust it? The corpus folder labels are
// wrong often enough that "score unverified" must be visible -- a mismatch there
// is a labeling failure, not a follower failure.
function scoreTag(c) {{
  if (c.score_source !== 'piece_id') return '<b style="color:#e0a54a">SCORE UNVERIFIED (folder label; piece-ID abstained)</b>';
  return '<b style="color:#5ac07a">score identified from audio</b>' +
         (c.relabeled ? ' <b style="color:#e0a54a">(RE-LABELED off the folder label)</b>' : '');
}}

function renderList() {{
  document.getElementById('clip-list').innerHTML = CLIPS.map((c,i)=>{{
    const done = c.existing || (i===cur && verdict);
    const conf = c.v1_confidence==null ? '?' : c.v1_confidence.toFixed(2);
    return `<div class="clip-item ${{i===cur?'active':''}} ${{done?'done':''}}" onclick="selectClip(${{i}})">
      <div class="cp">${{c.piece}}</div><div class="meta">${{c.video_id}} &middot; conf ${{conf}}</div></div>`;
  }}).join('');
}}

function decodeAt(tt, ss, t) {{
  if (!tt.length) return null;
  if (t <= tt[0]) return ss[0];
  if (t >= tt[tt.length-1]) return ss[ss.length-1];
  let lo=0, hi=tt.length-1;
  while (lo+1<hi) {{ const m=(lo+hi)>>1; if (tt[m] < t) lo=m; else hi=m; }}
  const at=tt[lo], bt=tt[hi], as=ss[lo], bs=ss[hi];
  return bt===at ? bs : as + (bs-as)*((t-at)/(bt-at));
}}

async function selectClip(i) {{
  // Leaving a clip with marks or a verdict that were never saved silently
  // destroyed them, which is how wrong_spans reached disk as []. Refuse instead.
  if (cur>=0 && i!==cur && dirty()) {{
    if (!confirm('This clip has '+wrongSpans.length+' unsaved mark(s)'+(verdict?' and verdict "'+verdict+'"':'')+'. Leave without saving?')) return;
  }}
  if (raf) cancelAnimationFrame(raf);
  if (audioEl) audioEl.pause();
  const c=CLIPS[i];
  cur=i; spanOpen=null;
  wrongSpans = (c.saved_spans||[]).map(s=>s.slice());
  verdict = c.saved_verdict || null;
  document.getElementById('panel').innerHTML = `<p class="meta">loading follower for ${{c.piece}}/${{c.video_id}}...</p>`;
  const r = await fetch(`/clip/${{c.piece}}/${{c.video_id}}`);
  view = await r.json();
  if (view.error) {{ document.getElementById('panel').innerHTML = `<p class="meta">error: ${{view.error}}</p>`; return; }}
  renderPanel(); renderList();
}}

function renderPanel() {{
  const c=CLIPS[cur];
  document.getElementById('panel').innerHTML = `
    <h2>${{view.score_id}}</h2>
    <div class="meta">${{scoreTag(c)}} &middot; filed under ${{c.piece}}</div>
    <div class="meta">${{c.title||''}} &middot; ${{c.video_id}} &middot; follower confidence ${{view.median_confidence==null?'?':view.median_confidence}} &middot; transpose ${{view.transpose_semitones}}</div>
    <canvas id="roll" width="1100" height="320"></canvas>
    <div class="legend"><span class="sw" style="background:#4a9ae0"></span>played (where follower placed it)
      <span class="sw" style="background:#888"></span>score (reference)
      <span class="sw" style="background:#e05a5a"></span>played note whose pitch isn't in the score here</div>
    <audio id="audio" controls preload="auto" src="/audio/${{c.piece}}/${{c.video_id}}"></audio>
    <div class="wrongbar" id="wrongbar"></div>
    <div class="hint">Hold <kbd>Space</kbd> while the two rows clearly disagree under the green playhead. <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> seek 3s.</div>
    <div class="verdicts">
      ${{[['tracked','Tracked throughout'],['recovered','Lost, then recovered'],['wrong','Drifted / wrong'],['junk','Junk clip (right to give up)']].map(
        ([k,l])=>`<button class="verdict-btn" data-v="${{k}}" onclick="setVerdict('${{k}}')">${{l}}</button>`).join('')}}
    </div>
    <div class="save-row"><button class="save-btn" onclick="saveClip()">Save</button>
      <span class="status" id="markcount"></span>
      <span class="status" id="status"></span></div>`;
  audioEl=document.getElementById('audio');
  canvas=document.getElementById('roll'); ctx=canvas.getContext('2d');
  audioEl.addEventListener('timeupdate', updateWrongBar);
  // Space is the mark key. If the audio element holds focus it also toggles
  // native play/pause, which freezes currentTime under the hold and destroys
  // the mark -- so never let the player keep focus.
  audioEl.addEventListener('focus', ()=>audioEl.blur());
  audioEl.addEventListener('loadedmetadata', ()=>{{ updateWrongBar(); updateMarkCount(); }});
  syncVerdictButtons(); updateWrongBar(); updateMarkCount();
  loop();
}}

function pitchY(p, h) {{ return h - ((p-PITCH_MIN)/(PITCH_MAX-PITCH_MIN))*h; }}

function loop() {{
  raf=requestAnimationFrame(loop);
  if (!ctx||!view) return;
  const w=canvas.width, h=canvas.height, mid=h/2;
  ctx.clearRect(0,0,w,h);
  const t = audioEl ? audioEl.currentTime : 0;
  const curS = decodeAt(view.traj_t, view.traj_s, t);   // follower's score pos now
  if (curS==null) return;
  const cx=w/2;
  // playhead
  ctx.strokeStyle='#4ade80'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,h); ctx.stroke();
  ctx.strokeStyle='#222'; ctx.beginPath(); ctx.moveTo(0,mid); ctx.lineTo(w,mid); ctx.stroke();
  // pitches present in the score within a small window around the playhead (for match test)
  const near = new Set(view.score.filter(n=>Math.abs(n.sx-curS)<0.6).map(n=>n.pitch));
  const draw=(notes,yTop,yBot,colorFn)=>{{
    for (const n of notes) {{
      const x = cx + (n.sx-curS)*PX_PER_SCORE_SEC;
      if (x< -5 || x>w+5) continue;
      const y = yTop + (pitchY(n.pitch,yBot-yTop));
      ctx.fillStyle=colorFn(n); ctx.fillRect(x-2, y-2, 5, 4);
    }}
  }};
  // TOP: played notes (mapped by follower). red if pitch not in the score near the playhead AND near the playhead.
  draw(view.played, 6, mid-6, (n)=> (Math.abs(n.sx-curS)<0.6 && !near.has(n.pitch)) ? '#e05a5a' : '#4a9ae0');
  // BOTTOM: score reference
  draw(view.score, mid+6, h-6, ()=> '#888');
}}

// Unsaved work = marks or a verdict that differ from what is on disk for this clip.
function dirty() {{
  if (cur<0) return false;
  const c=CLIPS[cur];
  return JSON.stringify(wrongSpans)!==JSON.stringify((c.saved_spans||[]).map(s=>s.slice()))
      || verdict!==(c.saved_verdict||null);
}}
function setVerdict(v) {{ verdict=v; syncVerdictButtons(); updateMarkCount(); }}
function syncVerdictButtons() {{
  document.querySelectorAll('.verdict-btn').forEach(b=>b.classList.toggle('sel',b.dataset.v===verdict));
}}
// The mark count is shown next to Save so what will be POSTed is always visible.
// Silent divergence between the red bar and the payload is the failure we hit.
function updateMarkCount() {{
  const el=document.getElementById('markcount'); if(!el) return;
  el.textContent = wrongSpans.length===0 ? 'no marks' :
    wrongSpans.length+' mark'+(wrongSpans.length===1?'':'s')+
    ' ('+wrongSpans.reduce((s,[a,b])=>s+(b-a),0).toFixed(1)+'s)';
  el.style.color = wrongSpans.length ? '#e05a5a' : '#666';
}}

function updateWrongBar() {{
  if (!audioEl||!audioEl.duration) return;
  const bar=document.getElementById('wrongbar'); if(!bar) return;
  const D=audioEl.duration;
  const segs = wrongSpans.map(([a,b])=>`<div class="seg" style="left:${{100*a/D}}%;width:${{100*(b-a)/D}}%"></div>`).join('');
  const ph=`<div class="ph" style="left:${{100*audioEl.currentTime/D}}%"></div>`;
  bar.innerHTML=segs+ph;
}}

// A reload also discards in-memory marks; make that a prompt, not a surprise.
window.addEventListener('beforeunload',(e)=>{{ if (dirty()) {{ e.preventDefault(); e.returnValue=''; }} }});

document.addEventListener('keydown',(e)=>{{
  if (e.target.tagName==='INPUT') return;
  if (e.code==='Space') {{ e.preventDefault(); if(spanOpen==null && audioEl) spanOpen=audioEl.currentTime; }}
  else if (e.code==='ArrowLeft' && audioEl) {{ e.preventDefault(); audioEl.currentTime=Math.max(0,audioEl.currentTime-3); }}
  else if (e.code==='ArrowRight' && audioEl) {{ e.preventDefault(); audioEl.currentTime+=3; }}
}});
document.addEventListener('keyup',(e)=>{{
  if (e.code==='Space' && spanOpen!=null && audioEl) {{
    e.preventDefault();
    const a=Math.min(spanOpen,audioEl.currentTime), b=Math.max(spanOpen,audioEl.currentTime);
    // A hold that spans no playback time used to vanish with no feedback. Say so:
    // it means the audio was paused under the hold, and the mark was NOT recorded.
    if (b-a>0.05) {{ wrongSpans.push([a,b]); }}
    else {{
      const st=document.getElementById('status');
      if (st) st.textContent='mark NOT recorded: audio did not advance under the hold (paused?) -- play, then hold Space';
    }}
    spanOpen=null; updateWrongBar(); updateMarkCount();
  }}
}});

function saveClip() {{
  const c=CLIPS[cur], st=document.getElementById('status');
  if (!verdict) {{ st.textContent='pick a verdict first'; return; }}
  const D = audioEl && audioEl.duration ? audioEl.duration : null;
  const wrong = wrongSpans.reduce((s,[a,b])=>s+(b-a),0);
  // The red bar and the payload must agree. If they ever diverge, stop -- do not
  // write a record whose spans contradict what the labeler was looking at.
  const shown = document.querySelectorAll('#wrongbar .seg').length;
  if (shown !== wrongSpans.length) {{
    st.textContent='NOT saved: '+shown+' mark(s) on the bar but '+wrongSpans.length+' in memory. Reload and re-mark this clip.';
    return;
  }}
  if ((verdict==='wrong'||verdict==='junk') && wrongSpans.length===0
      && !confirm('Verdict "'+verdict+'" with no marked spans. Save anyway?')) {{ return; }}
  st.textContent='saving...';
  fetch('/save-validate',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{piece:c.piece,video_id:c.video_id,verdict,wrong_spans:wrongSpans,
      score_id:view.score_id,score_source:view.score_source,
      follower_confidence:view.median_confidence,
      audio_duration_sec:D, fraction_wrong: D? +(wrong/D).toFixed(4): null}})}})
  .then(r=>r.json()).then(d=>{{ if(d.error) throw new Error(d.error);
    CLIPS[cur].existing=true;
    // Track what is now on disk so dirty() and re-select restore stay accurate.
    CLIPS[cur].saved_spans=wrongSpans.map(s=>s.slice()); CLIPS[cur].saved_verdict=verdict;
    st.textContent='saved '+wrongSpans.length+' mark(s) -> '+d.path; renderList(); }})
  .catch(e=>{{ st.textContent='error: '+e.message; }});
}}

renderList();
</script></body></html>"""


# ---------------------------------------------------------------------------
# HTTP server (Range WAV + lazy per-clip follower compute)
# ---------------------------------------------------------------------------


class ValidateHandler(http.server.BaseHTTPRequestHandler):
    clips: list[dict]
    bundles_root: Path
    scores_root: Path
    _wav_by_key: dict
    _clip_by_key: dict
    _view_cache: dict

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = generate_html(self.clips).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path.startswith("/clip/"):
            self._serve_clip(self.path[len("/clip/") :])
        elif self.path.startswith("/audio/"):
            self._serve_wav(self.path[len("/audio/") :])
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_clip(self, key: str):
        if key in self._view_cache:
            self._json(self._view_cache[key])
            return
        try:
            piece, vid = key.split("/", 1)
            c = self._clip_by_key[key]
            view = get_clip_view(
                piece,
                vid,
                self.bundles_root,
                self.scores_root,
                c["score_id"],
                c["score_source"],
            )
            self._view_cache[key] = view
            self._json(view)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, code=400)

    def _serve_wav(self, key: str):
        wav = self._wav_by_key.get(key)
        if wav is None or not wav.exists():
            self.send_response(404)
            self.end_headers()
            return
        size = wav.stat().st_size
        rng = self.headers.get("Range")
        start, end, partial = 0, size - 1, False
        if rng and rng.startswith("bytes="):
            partial = True
            s, _, e = rng[len("bytes=") :].partition("-")
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
                    return
                remaining -= len(chunk)

    def do_POST(self):
        if self.path != "/save-validate":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n))
            key = f"{payload['piece']}/{payload['video_id']}"
            if key not in self._clip_by_key:
                raise ValidateToolError(f"clip {key} is not in this validation session")
            clip = self._clip_by_key[key]
            view = get_clip_view(
                clip["piece"],
                clip["video_id"],
                self.bundles_root,
                self.scores_root,
                clip["score_id"],
                clip["score_source"],
            )
            payload["score_id"] = view["score_id"]
            payload["score_source"] = view["score_source"]
            payload["follower_confidence"] = view["median_confidence"]
            path = save_validation(self.bundles_root, payload)
            self._json({"ok": True, "path": path.name})
        except Exception as e:
            self._json({"error": str(e)}, code=400)

    def log_message(self, format, *args):  # noqa: A002
        pass


def make_handler(clips, bundles_root, scores_root):
    wav_by_key = {f"{c['piece']}/{c['video_id']}": c["wav_path"] for c in clips}

    class Bound(ValidateHandler):
        pass

    Bound.clips = clips
    Bound.bundles_root = bundles_root
    Bound.scores_root = scores_root
    Bound._wav_by_key = wav_by_key
    Bound._clip_by_key = {f"{c['piece']}/{c['video_id']}": c for c in clips}
    Bound._view_cache = {}
    return Bound


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Light-touch follower validator (#133 Track B)"
    )
    ap.add_argument("--subset", type=Path, default=SUBSET_JSON)
    ap.add_argument(
        "--bundles-root", type=Path, default=Path("data/evals/realaudio_bundles")
    )
    ap.add_argument("--scores-root", type=Path, default=Path("data/scores"))
    ap.add_argument(
        "--all",
        action="store_true",
        help="validate every bundle, not just the gold subset",
    )
    ap.add_argument("--pieces", nargs="+", default=None)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument(
        "--precompute",
        action="store_true",
        help="run the follower on every clip and cache the views to disk, then exit "
        "(do this once -- big clips take minutes -- so the labeler loads instantly)",
    )
    ap.add_argument(
        "--force", action="store_true", help="with --precompute: rebuild cached views"
    )
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument(
        "--piece-id",
        type=Path,
        default=None,
        help="piece-ID results JSON (default <bundles-root>/_piece_id.json); "
        "decides which score each clip is validated against",
    )
    ap.add_argument(
        "--trust-labels",
        action="store_true",
        help="validate against the corpus folder label instead of a piece-ID run "
        "(the labels are known-wrong -- only for a label-free smoke test)",
    )
    args = ap.parse_args()

    id_map: dict[str, str] = {}
    if not args.trust_labels:
        id_map = load_piece_id_map(
            args.piece_id or args.bundles_root / "_piece_id.json"
        )

    clips = list_clips(args.subset, args.bundles_root, args.all, args.pieces, id_map)
    n_id = sum(1 for c in clips if c["score_source"] == "piece_id")
    n_relabel = sum(1 for c in clips if c["relabeled"])
    print(
        f"Resolved {len(clips)} clips ({sum(1 for c in clips if c['existing'])} already validated)"
    )
    print(
        f"  score source: {n_id} identified ({n_relabel} RE-LABELED off the folder label), "
        f"{len(clips) - n_id} falling back to the unverified label"
    )

    if args.precompute:
        import time

        for i, c in enumerate(clips, 1):
            key = f"{c['piece']}/{c['video_id']}"
            cache = _view_cache_path(
                args.bundles_root, c["piece"], c["video_id"], c["score_id"]
            )
            if cache.exists() and not args.force:
                print(f"[{i}/{len(clips)}] {key}: cached", flush=True)
                continue
            t0 = time.perf_counter()
            try:
                get_clip_view(
                    c["piece"],
                    c["video_id"],
                    args.bundles_root,
                    args.scores_root,
                    c["score_id"],
                    c["score_source"],
                    force=args.force,
                )
                print(
                    f"[{i}/{len(clips)}] {key}: built in {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[{i}/{len(clips)}] {key}: FAILED {type(e).__name__}: {e}",
                    flush=True,
                )
        return

    if not args.serve:
        for c in clips[:10]:
            conf = "?" if c["v1_confidence"] is None else f"{c['v1_confidence']:.2f}"
            print(f"  {c['piece']}/{c['video_id']}  conf={conf}")
        print("  ... run with --precompute (once), then --serve to validate")
        return
    handler = make_handler(clips, args.bundles_root, args.scores_root)
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        httpd.daemon_threads = True
        print(f"Validator: http://localhost:{args.port}   (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    sys.exit(main())
