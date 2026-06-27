"""Axis A: real-audio -> production AMT -> piece-ID gate (the #1 fidelity gap).

The cross-performance and comprehensive harnesses feed CLEAN performance MIDIs to
the gate. Production never sees clean MIDI: it sees microphone audio transcribed
by the Aria-AMT service (noisy -- missed/extra notes, smeared onsets, octave
errors). This harness measures the recognition drop on the REAL path:

    MAESTRO audio (the same rendition as an ASAP perf MIDI)
      -> slice into 15s chunks with 15s context overlap   (production chunking)
      -> POST each to the local AMT service (apps/inference/amt, :8001)
      -> stitch deduplicated notes                          (SessionBrain buffer)
      -> the FROZEN piece-ID gate (chroma top-K -> elastic margin -> lock)

PAIRED design: 519/1066 ASAP perfs carry a `maestro_audio_performance` path, so
for each we have BOTH the clean ASAP MIDI and the MAESTRO audio of the SAME
performance. We run the identical gate on both and report the paired recognition
delta -- isolating the AMT-noise penalty from every other confound.

GATED on (fail loud, never silently skip -- explicit-exception policy):
  * MAESTRO audio rehydrated (offloaded ~34GB; --maestro-dir points at the root
    that the metadata.csv `{maestro}` placeholder resolves to). A sample subset
    of the 519 matched performances is enough for a pilot (--limit).
  * The AMT service healthy at --amt-url (run `just amt`; model load is slow).

Run (after MAESTRO is rehydrated + `just amt` is up):
  PYTHONPATH=src uv run python -m score_library.pieceid_amt_axis \
      --scores-dir /path/to/issue-49/model/data/scores \
      --maestro-dir /path/to/maestro-v3.0.0 \
      --amt-url http://localhost:8001 --limit 50 --opening-seconds 90
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from piece_id_eval.notes import Note
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate
from score_library.bulk_ingest import TOP_K, _RunningChromaIndex
from score_library.pieceid_comprehensive_eval import (
    DEFAULT_THRESHOLD,
    load_catalog,
    recognized_at,
)
from score_library.pieceid_crossperf_verify import (
    _ASAP_DIR,
    _SCORES_DIR,
    _load_perf_midi,
    eval_query,
    label_works,
)

SAMPLE_RATE = 16000          # AMT decodes everything to 16 kHz mono
CHUNK_SECONDS = 15.0         # production chunk length
CONTEXT_SECONDS = 15.0       # previous-chunk overlap fed as context


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Audio -> AMT notes (production-faithful chunking)
# ---------------------------------------------------------------------------

def _wav_bytes(pcm: np.ndarray) -> bytes:
    """Encode a float32 mono PCM array as 16-bit WAV bytes (ffmpeg-decodable)."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _amt_transcribe_chunk(amt_url: str, chunk_pcm: np.ndarray,
                          context_pcm: np.ndarray | None) -> list[dict]:
    """POST one 15s chunk (+ optional context) to the AMT service; return its
    midi_notes (already de-duplicated to the current chunk by the handler)."""
    payload = {"chunk_audio": base64.b64encode(_wav_bytes(chunk_pcm)).decode()}
    if context_pcm is not None and len(context_pcm):
        payload["context_audio"] = base64.b64encode(_wav_bytes(context_pcm)).decode()
    req = urllib.request.Request(
        amt_url.rstrip("/") + "/transcribe",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    if "error" in body:
        raise RuntimeError(f"AMT error: {body['error']}")
    return body["midi_notes"]


def audio_to_notes(amt_url: str, audio_path: Path, opening_seconds: float) -> list[Note]:
    """Transcribe the opening `opening_seconds` of an audio file through the AMT
    service using production 15s-chunk + 15s-context windowing, and stitch the
    de-duplicated notes onto an absolute timeline."""
    import librosa
    pcm, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    pcm = pcm[: int(opening_seconds * SAMPLE_RATE)]
    chunk_n = int(CHUNK_SECONDS * SAMPLE_RATE)
    ctx_n = int(CONTEXT_SECONDS * SAMPLE_RATE)

    notes: list[Note] = []
    pos = 0
    while pos < len(pcm):
        chunk = pcm[pos: pos + chunk_n]
        if len(chunk) < SAMPLE_RATE:  # <1s tail: nothing useful to transcribe
            break
        context = pcm[max(0, pos - ctx_n): pos] if pos > 0 else None
        chunk_notes = _amt_transcribe_chunk(amt_url, chunk, context)
        t_offset = pos / SAMPLE_RATE   # de-duped notes are chunk-relative
        for nd in chunk_notes:
            notes.append(Note(
                onset=float(nd["onset"]) + t_offset,
                offset=float(nd["offset"]) + t_offset,
                pitch=int(nd["pitch"]),
                velocity=int(nd.get("velocity", 80)),
            ))
        pos += chunk_n
    notes.sort(key=lambda n: n.onset)
    return notes


# ---------------------------------------------------------------------------
# MAESTRO audio mapping
# ---------------------------------------------------------------------------

def _maestro_rel(rel: str) -> str:
    """metadata.csv stores `{maestro}/2006/...wav`; strip the placeholder ->
    the year/filename path used as both the local-dir suffix and the HF repo path."""
    rel = rel.strip()
    if rel.startswith("{maestro}"):
        rel = rel[len("{maestro}"):].lstrip("/")
    return rel


def matched_performances(maestro_dir: Path | None, limit: int,
                         hf_repo: str | None) -> list[dict]:
    """ASAP folders with a `maestro_audio_performance`. In HF-stream mode the audio
    is fetched per-perf on demand (no local existence check, so the full 519 fit in
    a few GB of disk); in local-dir mode only perfs whose file exists are kept."""
    meta = _ASAP_DIR / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"ASAP metadata.csv missing: {meta}")
    rows: list[dict] = []
    with meta.open() as fh:
        for row in csv.DictReader(fh):
            rel = row.get("maestro_audio_performance", "").strip()
            if not rel:
                continue
            hf_path = _maestro_rel(rel)
            entry = {"folder": row["folder"], "hf_path": hf_path,
                     "perf_midi": _ASAP_DIR / row["midi_performance"]}
            if hf_repo:
                rows.append(entry)            # fetch on demand later
            else:
                if maestro_dir is None:
                    raise ValueError("maestro_dir required when not streaming")
                audio = maestro_dir / hf_path
                if audio.exists():
                    entry["audio"] = audio
                    rows.append(entry)
    if not rows:
        raise FileNotFoundError(
            f"No MAESTRO-matched performances. In local mode rehydrate MAESTRO v3 "
            f"under {maestro_dir}; or stream with --stream-hf ddPn08/maestro-v3.0.0.")
    if limit:
        rows = rows[:limit]
    return rows


def _fetch_hf_audio(hf_repo: str, hf_path: str, dest_dir: Path) -> Path:
    """Download one MAESTRO WAV from an HF dataset to dest_dir (disk-bounded streaming)."""
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=hf_repo, filename=hf_path, repo_type="dataset",
                            local_dir=str(dest_dir))
    return Path(local)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _check_amt(amt_url: str) -> None:
    try:
        with urllib.request.urlopen(amt_url.rstrip("/") + "/health", timeout=10) as r:
            h = json.loads(r.read())
    except Exception as e:
        raise RuntimeError(
            f"AMT service not reachable at {amt_url} ({e}). Start it: `just amt` "
            f"(model load is slow; wait for '[AMT] Model loaded. Ready.').")
    if not h.get("loaded"):
        raise RuntimeError(f"AMT service up but model not loaded: {h}")
    _log(f"AMT healthy: {h}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", type=str, default=str(_SCORES_DIR))
    ap.add_argument("--maestro-dir", type=str, default="",
                    help="local MAESTRO v3 root (the {maestro} target); omit when streaming")
    ap.add_argument("--stream-hf", type=str, default="",
                    help="HF dataset id to stream audio per-perf from (e.g. ddPn08/maestro-v3.0.0); "
                         "downloads each WAV on demand and deletes it -> disk stays ~2-3GB for the full 519")
    ap.add_argument("--keep-audio", action="store_true",
                    help="in stream mode, keep downloaded WAVs instead of deleting after transcription")
    ap.add_argument("--amt-url", type=str, default="http://localhost:8001")
    ap.add_argument("--limit", type=int, default=50,
                    help="number of MAESTRO-matched performances (0 = all 519)")
    ap.add_argument("--opening-seconds", type=float, default=90.0)
    ap.add_argument("--note-cap", type=int, default=600)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--catalog-cache", type=str, default="",
                    help="parsed-catalog pickle cache (shared with the comprehensive eval)")
    ap.add_argument("--out", type=str,
                    default=str(_ASAP_DIR.parent.parent / "evals" / "piece_id" / "amt_axis.json"))
    args = ap.parse_args()

    if not args.stream_hf and not args.maestro_dir:
        raise SystemExit("provide --maestro-dir (local audio) or --stream-hf <repo> (per-perf download)")

    t0 = time.time()
    _check_amt(args.amt_url)
    maestro_dir = Path(args.maestro_dir) if args.maestro_dir else None
    rows = matched_performances(maestro_dir, args.limit, args.stream_hf or None)
    _log(f"{len(rows)} MAESTRO-matched performances"
         + (f" (streaming from {args.stream_hf})" if args.stream_hf else ""))

    _log(f"building catalog from {args.scores_dir} ...")
    catalog = load_catalog(Path(args.scores_dir), args.note_cap,
                           args.catalog_cache or None, exclude=None)
    _log(f"  catalog: {len(catalog)} pieces  [{time.time()-t0:.1f}s]")
    chroma = _RunningChromaIndex(catalog)
    gate = ElasticGate(catalog)
    labels, _ = label_works(catalog, chroma, gate, args.note_cap)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # Resume: keep prior records, skip folders already processed (overnight resilience).
    records: list[dict] = []
    done: set[str] = set()
    if outp.exists():
        try:
            prior = json.loads(outp.read_text()).get("records", [])
            records = prior
            done = {r["work"] for r in prior}
            _log(f"resuming: {len(done)} performances already processed")
        except Exception:
            pass

    stream_dir = outp.parent / "_maestro_stream"
    for i, r in enumerate(rows):
        folder = r["folder"]
        if folder not in labels or folder in done:
            continue
        true_id = labels[folder]["true_id"]
        # Clean-MIDI baseline (same performance).
        clean_notes = _load_perf_midi(r["perf_midi"])[: args.note_cap]
        clean = eval_query(clean_notes, true_id, chroma, gate, args.top_k)
        # Real-audio -> AMT path. Stream the WAV on demand, then delete it.
        audio_path = r.get("audio")
        fetched = False
        try:
            if args.stream_hf and audio_path is None:
                audio_path = _fetch_hf_audio(args.stream_hf, r["hf_path"], stream_dir)
                fetched = True
            if audio_path is None:
                raise FileNotFoundError(f"no audio resolved for {folder}")
            amt_notes = audio_to_notes(args.amt_url, audio_path, args.opening_seconds)
        except Exception as e:
            _log(f"  [{folder}] AMT/fetch failed: {e}")
            continue
        finally:
            if fetched and not args.keep_audio and audio_path and Path(audio_path).exists():
                Path(audio_path).unlink()
        amt_capped = amt_notes[: args.note_cap]
        amt = eval_query(amt_capped, true_id, chroma, gate, args.top_k) if len(amt_capped) >= 2 else None
        rec = {
            "work": folder, "true_id": true_id,
            "clean_n": len(clean_notes), "amt_n": len(amt_notes),
            "clean_recognized": recognized_at(clean, args.threshold) if clean else None,
            "clean_recall_at_k": clean.get("chroma_recall_at_k") if clean else None,
            "clean_margin": clean.get("closed_margin") if clean else None,
            "amt_recognized": recognized_at(amt, args.threshold) if amt else None,
            "amt_recall_at_k": amt.get("chroma_recall_at_k") if amt else None,
            "amt_margin": amt.get("closed_margin") if amt else None,
            "amt_best": amt.get("closed_best") if amt else None,
        }
        records.append(rec)
        # Persist after every performance so an overnight interruption resumes.
        outp.write_text(json.dumps({"records": records}, indent=2))
        if (i + 1) % 10 == 0:
            _log(f"  [{i+1}/{len(rows)}] processed, {len(records)} records  [{time.time()-t0:.1f}s]")

    # Paired aggregation.
    paired = [r for r in records if r["clean_recognized"] is not None and r["amt_recognized"] is not None]
    n = len(paired)
    def _frac(key):
        return round(sum(1 for r in paired if r[key]) / n, 4) if n else None
    summary = {
        "n_paired": n,
        "clean_recognized": _frac("clean_recognized"),
        "amt_recognized": _frac("amt_recognized"),
        "clean_recall_at_k": _frac("clean_recall_at_k"),
        "amt_recall_at_k": _frac("amt_recall_at_k"),
        "recognition_drop": round((_frac("clean_recognized") or 0) - (_frac("amt_recognized") or 0), 4) if n else None,
        "both_recognized": round(sum(1 for r in paired if r["clean_recognized"] and r["amt_recognized"]) / n, 4) if n else None,
        "clean_only": round(sum(1 for r in paired if r["clean_recognized"] and not r["amt_recognized"]) / n, 4) if n else None,
        "amt_note_yield_median": round(float(np.median([r["amt_n"] for r in paired])), 1) if n else None,
        "clean_note_median": round(float(np.median([r["clean_n"] for r in paired])), 1) if n else None,
    }
    out = {
        "config": {"limit": args.limit, "opening_seconds": args.opening_seconds,
                   "note_cap": args.note_cap, "threshold": args.threshold,
                   "catalog_size": len(catalog), "amt_url": args.amt_url},
        "summary": summary,
        "records": records,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    print(f"AXIS A -- REAL-AUDIO -> AMT -> GATE  (paired n={n}, catalog={len(catalog)})")
    print("=" * 72)
    print(f"  clean-MIDI recognized: {_pct(summary['clean_recognized'])}   "
          f"AMT-audio recognized: {_pct(summary['amt_recognized'])}")
    print(f"  RECOGNITION DROP (clean - amt): {_pct(summary['recognition_drop'])}")
    print(f"  chroma recall@k  clean: {_pct(summary['clean_recall_at_k'])}   "
          f"amt: {_pct(summary['amt_recall_at_k'])}")
    print(f"  AMT note yield (median): {summary['amt_note_yield_median']} vs clean {summary['clean_note_median']}")
    print(f"\nwrote {outp}   [total {time.time()-t0:.1f}s]")


def _pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
