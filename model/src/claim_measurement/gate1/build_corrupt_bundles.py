"""GATE 1 runner (model env): produce corrupt measurement bundles + warp maps.

For each scored cached clip and each corruption spec, degrade the clean audio with
a construction-known transform, re-run the FULL pipeline (real Aria-AMT + parangonar)
to produce a corrupt bundle, and persist the bundle alongside a meta file carrying
the warp map and a pointer to the clean bundle. The downstream apps/evals analyzer
(claim_taxonomy.gate1) consumes these to compute per-bar localization error.

This is the slow, AMT-calling half of GATE 1. The pure spec->audio dispatch
(`build_corrupted_audio`) is unit-tested; `main()` is run manually against a live
AMT server. No LLM is involved.

Usage:
    AMT_URL=http://127.0.0.1:8001/transcribe \
        uv run python -m claim_measurement.gate1.build_corrupt_bundles [--limit-piece bach_invention_1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from chroma_dtw_eval.amt_regen import (
    DEFAULT_AMT_URL,
    DEFAULT_SCORE_BY_PIECE,
    TARGET_SR,
    _read_wav_16k_mono,
)

from claim_measurement.extractor import BundleExtractionError, extract_bundle
from claim_measurement.gate1.corruption import (
    WarpMap,
    add_noise,
    apply_piecewise_time_warp,
    pitch_shift_region,
    silence_region,
)

_MODULE_DIR = Path(__file__).resolve()
DEFAULT_DATA_ROOT = _MODULE_DIR.parents[3] / "data" / "evals"
DEFAULT_CLEAN_BUNDLE_ROOT = DEFAULT_DATA_ROOT / "claim_bundles"
DEFAULT_OUT_ROOT = DEFAULT_DATA_ROOT / "gate1"


@dataclass
class CorruptionSpec:
    kind: str
    params: dict = field(default_factory=dict)

    @property
    def spec_id(self) -> str:
        p = self.params
        if self.kind == "clean":
            return "clean"
        if self.kind == "tempo":
            return f"tempo_{p.get('label', 'warp')}"
        if self.kind == "noise":
            return f"noise_snr{int(p['snr_db'])}"
        if self.kind == "dropout":
            return "dropout"
        if self.kind == "pitch":
            return f"pitch_p{int(p['semitones'])}"
        return self.kind


def build_corrupted_audio(
    audio: np.ndarray, sr: int, spec: CorruptionSpec, rng: np.random.Generator
) -> tuple[np.ndarray, WarpMap]:
    """Dispatch a CorruptionSpec to its corruption + its construction-known warp map."""
    dur = len(audio) / sr
    if spec.kind == "clean":
        return audio.copy(), WarpMap.identity(dur)
    if spec.kind == "tempo":
        segs = [tuple(s) for s in spec.params["segments"]]
        return apply_piecewise_time_warp(audio, sr, segs)
    if spec.kind == "noise":
        return add_noise(audio, float(spec.params["snr_db"]), rng), WarpMap.identity(dur)
    if spec.kind == "dropout":
        out = silence_region(audio, sr, float(spec.params["start_sec"]), float(spec.params["end_sec"]))
        return out, WarpMap.identity(dur)
    if spec.kind == "pitch":
        out = pitch_shift_region(
            audio, sr,
            float(spec.params["start_sec"]), float(spec.params["end_sec"]),
            float(spec.params["semitones"]),
        )
        return out, WarpMap.identity(dur)
    raise ValueError(f"unknown corruption kind: {spec.kind}")


def default_sweep(duration_sec: float) -> list[CorruptionSpec]:
    """Corruption sweep scaled to a clip's duration. Tempo warps target the middle
    third; dropout/pitch target a shorter mid window."""
    lo, hi = 0.33 * duration_sec, 0.66 * duration_sec
    m_lo, m_hi = 0.40 * duration_sec, 0.52 * duration_sec
    return [
        CorruptionSpec("clean"),
        CorruptionSpec("tempo", {"segments": [[lo, hi, 1.3]], "label": "rush_1.3x"}),
        CorruptionSpec("tempo", {"segments": [[lo, hi, 0.77]], "label": "drag_0.77x"}),
        CorruptionSpec("tempo", {"segments": [[lo, hi, 1.6]], "label": "rush_1.6x"}),
        CorruptionSpec("noise", {"snr_db": 20.0}),
        CorruptionSpec("noise", {"snr_db": 10.0}),
        CorruptionSpec("noise", {"snr_db": 5.0}),
        CorruptionSpec("dropout", {"start_sec": m_lo, "end_sec": m_hi}),
        CorruptionSpec("pitch", {"start_sec": m_lo, "end_sec": m_hi, "semitones": 2.0}),
    ]


def _stable_seed(spec_id: str, base: int) -> int:
    h = hashlib.sha256(f"{base}:{spec_id}".encode()).digest()[:4]
    return int.from_bytes(h, "big")


@dataclass
class ClipSpec:
    piece_id: str
    video_id: str
    audio_path: Path
    score_path: Path


def _resolve_clips(clean_bundle_root: Path, score_by_piece: dict[str, Path]) -> list[ClipSpec]:
    """Enumerate clips from the existing clean bundles (the authoritative pairing).

    Each clean bundle carries the real audio_path; deriving clips from the bundles
    (rather than globbing practice_eval audio) avoids spurious/duplicate WAVs and
    guarantees every clip has a clean bundle to compare against.
    """
    clips: list[ClipSpec] = []
    for piece_id, score_path in score_by_piece.items():
        piece_dir = clean_bundle_root / piece_id
        if not piece_dir.is_dir():
            continue
        for bundle_path in sorted(piece_dir.glob("*.json")):
            if bundle_path.name == "_index.json":
                continue
            bundle = json.loads(bundle_path.read_text())
            clips.append(
                ClipSpec(piece_id, bundle_path.stem, Path(bundle["audio_path"]), score_path)
            )
    return clips


def run_clip(
    clip: ClipSpec,
    specs: list[CorruptionSpec],
    *,
    clean_bundle_root: Path,
    out_root: Path,
    amt_url: str,
    seed: int,
) -> list[dict]:
    """Corrupt+extract every spec for one clip. Returns one result row per spec."""
    audio = _read_wav_16k_mono(clip.audio_path)
    clean_bundle = clean_bundle_root / clip.piece_id / f"{clip.video_id}.json"
    if not clean_bundle.exists():
        raise BundleExtractionError(f"clean bundle missing for pairing: {clean_bundle}")

    rows: list[dict] = []
    for spec in specs:
        rng = np.random.default_rng(_stable_seed(spec.spec_id, seed))
        corrupt_audio, warp_map = build_corrupted_audio(audio, TARGET_SR, spec, rng)
        video_id = f"{clip.video_id}__{spec.spec_id}"
        base = {"piece": clip.piece_id, "video": clip.video_id, "spec_id": spec.spec_id}
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            sf.write(tf.name, corrupt_audio, TARGET_SR, subtype="FLOAT")
            try:
                bundle_path = extract_bundle(
                    clip.piece_id, video_id,
                    audio_path=Path(tf.name), score_path=clip.score_path,
                    cache_root=out_root, bundle_root=out_root,
                    amt_url=amt_url, force=True,
                )
            except BundleExtractionError as e:
                rows.append({**base, "status": "failed", "reason": str(e)})
                continue
        meta = {
            **base,
            "kind": spec.kind,
            "params": spec.params,
            "clean_bundle": str(clean_bundle.resolve()),
            "corrupt_bundle": str(bundle_path.resolve()),
            "warp_map": warp_map.to_dict(),
        }
        meta_path = out_root / clip.piece_id / f"{video_id}.meta.json"
        meta_path.write_text(json.dumps(meta))
        rows.append({**base, "status": "ok", "meta": str(meta_path)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim_measurement.gate1.build_corrupt_bundles")
    parser.add_argument("--clean-bundle-root", type=Path, default=DEFAULT_CLEAN_BUNDLE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--amt-url", default=DEFAULT_AMT_URL)
    parser.add_argument("--limit-piece", default=None)
    parser.add_argument("--limit-video", default=None,
                        help="If set, only extract this clip's video_id.")
    parser.add_argument("--only-specs", nargs="+", default=None,
                        help="If set, keep only these spec_ids (e.g. clean tempo_rush_1.3x noise_snr10).")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    clips = _resolve_clips(args.clean_bundle_root, dict(DEFAULT_SCORE_BY_PIECE))
    if args.limit_piece:
        clips = [c for c in clips if c.piece_id == args.limit_piece]
    if args.limit_video:
        clips = [c for c in clips if c.video_id == args.limit_video]
    if not clips:
        raise FileNotFoundError("no scored clips with audio found")

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for clip in clips:
        audio = _read_wav_16k_mono(clip.audio_path)
        specs = default_sweep(len(audio) / TARGET_SR)
        if args.only_specs:
            wanted = set(args.only_specs)
            specs = [s for s in specs if s.spec_id in wanted]
        rows = run_clip(
            clip, specs,
            clean_bundle_root=args.clean_bundle_root, out_root=args.out_root,
            amt_url=args.amt_url, seed=args.seed,
        )
        all_rows.extend(rows)
        for r in rows:
            tag = "OK  " if r["status"] == "ok" else "FAIL"
            line = f"  {tag} {r['piece']}/{r['video']} [{r['spec_id']}]"
            print(line if r["status"] == "ok" else line + f": {r.get('reason')}",
                  file=sys.stderr if r["status"] != "ok" else sys.stdout)

    index_path = args.out_root / "_index.json"
    index_path.write_text(json.dumps({"amt_url": args.amt_url, "seed": args.seed, "results": all_rows}, indent=2))
    ok = sum(1 for r in all_rows if r["status"] == "ok")
    print(json.dumps({"extracted": ok, "failed": len(all_rows) - ok, "index": str(index_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
