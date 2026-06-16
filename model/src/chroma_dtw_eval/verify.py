"""Verify CLI — practice-corpus + AMT-pseudo-truth path via committed manifest.

Contract:
  - stdout: exactly one float on a single line (the primary scalar).
  - exit: 0 iff no guard regressed; non-zero otherwise.
  - sidecar JSON: {primary, guards{g1,g2,g3,g4,g5}, baseline, regressed,
                   n_chunks, error_seconds_distribution, tolerance_sensitivity,
                   generated_at, g2_threshold_scale}.
  - stderr WARNING when manifest contains < 2 distinct pieces.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from pathlib import Path

from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, aggregate,
)
from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruth, load_pseudo_truth,
)

_MODULE_DIR = Path(__file__).resolve()
DEFAULT_SIDECAR = _MODULE_DIR.parents[2] / "data/evals/chroma_dtw/last_run.json"
DEFAULT_MANIFEST = (
    _MODULE_DIR.parents[2] / "data/evals/chroma_dtw_fixtures/manifest.json"
)
DEFAULT_AMT_VERSION_CONFIG = _MODULE_DIR.parents[2] / "config/amt_version.json"
DEFAULT_SCORE_BY_PIECE = {
    "bach_prelude_c_wtc1": _MODULE_DIR.parents[2] / "data/scores/bach.prelude.bwv_846.json",
    "bach_invention_1": _MODULE_DIR.parents[2] / "data/scores/bach.inventions.1.json",
}
DEFAULT_DECIM_HZ_SCORE_CHROMA = 50.0
TOLERANCE_SWEEP = (0.5, 1.0, 1.5, 2.0, 3.0)


def _sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _truth_score_audio_sec(pt: PseudoTruth, chunk_start_audio_sec: float) -> float:
    import numpy as np
    return float(np.interp(
        chunk_start_audio_sec, pt.perf_audio_sec, pt.score_audio_sec
    ))


def _build_results(
    manifest: list[dict],
    *,
    corpus_root: Path,
    pseudo_truth_root: Path,
    score_by_piece: dict[str, Path],
    checkpoint_hash: str,
    parangonar_version: str,
    decim_hz_score_chroma: float,
    skip_dtw: bool,
    band_back_s: float = 2.0,
    band_fwd_s: float = 12.0,
    use_band: bool = True,
    tempo_ratio: float = 0.45,
) -> list[ChunkResult]:
    from collections import defaultdict

    # Local-margin-in-band needs a per-chunk prior (expected score position).
    # The eval's chunks form a forward audio sweep within each clip, so we model
    # an online tracker: process each clip's chunks in audio order. The prior is
    # DEAD-RECKONED from the clip's first chunk -- expected score position grows at
    # a fixed `tempo_ratio` (score-seconds per audio-second); band-DTW then refines
    # the actual prediction within the window. This uses only audio timestamps (no
    # truth leak) and, unlike carrying the follower's own prediction forward, has
    # forward pressure -- a sticky early endpoint on a self-similar opening can no
    # longer freeze the band at the origin (the lock-to-origin failure mode).
    by_clip: dict[tuple[str, str], list[dict]] = defaultdict(list)
    clip_order: list[tuple[str, str]] = []
    for entry in manifest:
        key = (entry["piece"], entry["video_id"])
        if key not in by_clip:
            clip_order.append(key)
        by_clip[key].append(entry)

    results: list[ChunkResult] = []
    for key in clip_order:
        entries = sorted(by_clip[key], key=lambda e: float(e["start_audio_sec"]))
        clip_start_audio_sec = float(entries[0]["start_audio_sec"])
        # Per-clip ADAPTIVE tempo, seeded at tempo_ratio. After each chunk whose
        # implied cumulative tempo (pred / elapsed) is physically plausible
        # ([0.3, 1.0]), re-estimate tempo_est from it -- so a fast clip (a Bach
        # invention ~0.70) and a slow one (~0.50) each converge to their own rate,
        # which a single global tempo cannot serve. The plausible-range gate rejects
        # a stuck early prediction (pred~0 -> tempo 0) so it cannot poison the
        # estimate; the prior stays dead-reckoned (never set to a raw pred).
        tempo_est = tempo_ratio
        for entry in entries:
            piece = entry["piece"]
            video_id = entry["video_id"]
            start_audio_sec = float(entry["start_audio_sec"])
            end_audio_sec = float(entry["end_audio_sec"])
            audio_sha256 = entry["audio_sha256"]
            score_path = score_by_piece.get(piece)
            if score_path is None:
                raise FileNotFoundError(
                    f"no score JSON registered for piece {piece!r}; "
                    f"add to DEFAULT_SCORE_BY_PIECE in verify.py"
                )
            score_sha256 = _sha256_file(score_path)
            pt = load_pseudo_truth(
                piece_id=piece, video_id=video_id,
                audio_sha256=audio_sha256,
                amt_checkpoint_hash=checkpoint_hash,
                score_sha256=score_sha256,
                parangonar_version=parangonar_version,
                cache_root=pseudo_truth_root,
            )

            # The CLI reports the predicted score position at the chunk MIDPOINT
            # (score_frame_per_audio_frame[mid_audio]), so truth must be sampled
            # at the same audio time. Sampling at start_audio_sec instead
            # introduced a systematic ~half-chunk offset (~7.5s for a 15s chunk)
            # that drove the primary metric to ~0 even for a perfect follower.
            mid_audio_sec = 0.5 * (start_audio_sec + end_audio_sec)
            truth_score_sec = _truth_score_audio_sec(pt, mid_audio_sec)
            dead_reckon_residual_sec: float | None = None
            if skip_dtw:
                # Smoke path: predicted == truth (synthetic). Exercises sampler +
                # pseudo-truth + aggregator + sidecar without depending on DTW.
                predicted_score_sec = truth_score_sec
                cost = 0.1
            else:
                from chroma_dtw_eval.chroma_cache import ChromaParams, get_chroma
                from chroma_dtw_eval.dtw_runner import run_dtw
                audio_path = (
                    corpus_root / "practice_eval" / piece / "audio" / f"{video_id}.wav"
                )
                params = ChromaParams(target_frame_rate_hz=decim_hz_score_chroma, sr=16000)
                chroma = get_chroma(audio_path, params, corpus_root / "chroma_cache")
                start_f = int(round(start_audio_sec * chroma.frame_rate_hz))
                end_f = start_f + int(round(
                    (end_audio_sec - start_audio_sec) * chroma.frame_rate_hz
                ))
                seg = chroma.data[:, start_f:end_f].copy()
                if use_band:
                    # Dead-reckoned prior: expected score frame at this chunk's
                    # start = elapsed audio since clip start * tempo_ratio.
                    elapsed_audio_sec = start_audio_sec - clip_start_audio_sec
                    prior = int(round(
                        elapsed_audio_sec * tempo_est * chroma.frame_rate_hz
                    ))
                    band_back_frames = int(round(band_back_s * chroma.frame_rate_hz))
                    band_fwd_frames = int(round(band_fwd_s * chroma.frame_rate_hz))
                else:
                    prior, band_back_frames, band_fwd_frames = -1, 0, 0
                dtw = run_dtw(
                    seg, score_path,
                    frame_rate_hz=chroma.frame_rate_hz,
                    decim_hz=decim_hz_score_chroma,
                    prior_score_frame=prior,
                    band_back_frames=band_back_frames,
                    band_fwd_frames=band_fwd_frames,
                )
                predicted_score_sec = (
                    dtw.predicted_score_frame * (1.0 / decim_hz_score_chroma)
                )
                cost = float(dtw.cost)
                if use_band:
                    # g2 confidence signal: how far the DTW pulled the endpoint
                    # from the dead-reckon prior (prior is in score frames).
                    prior_sec = prior / chroma.frame_rate_hz
                    dead_reckon_residual_sec = abs(predicted_score_sec - prior_sec)
                # Adapt the clip tempo from this chunk's cumulative implied tempo
                # (pred / elapsed-to-midpoint), gated to a plausible range so a
                # stuck/teleported prediction cannot poison the estimate.
                mid_elapsed_sec = mid_audio_sec - clip_start_audio_sec
                if mid_elapsed_sec > 0.0:
                    obs_tempo = predicted_score_sec / mid_elapsed_sec
                    if 0.3 <= obs_tempo <= 1.0:
                        tempo_est = obs_tempo
            error_seconds = predicted_score_sec - truth_score_sec
            results.append(ChunkResult(
                kind="practice",
                piece=piece, video_id=video_id,
                start_audio_sec=start_audio_sec,
                predicted_score_sec=predicted_score_sec,
                error_seconds=error_seconds,
                cost=cost,
                dead_reckon_residual_sec=dead_reckon_residual_sec,
            ))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.verify")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=None,
                        help="Root containing practice_eval/, pseudo_truth/, scores/")
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--config", type=Path, default=DEFAULT_AMT_VERSION_CONFIG)
    parser.add_argument("--tolerance-s", type=float, default=1.5)
    parser.add_argument("--decim-hz", type=float, default=DEFAULT_DECIM_HZ_SCORE_CHROMA)
    parser.add_argument("--band-back-s", type=float, default=2.0,
                        help="local-margin band backward half-width (score seconds)")
    parser.add_argument("--band-fwd-s", type=float, default=12.0,
                        help="local-margin band forward half-width (score seconds); "
                             "negative disables the band (global endpoint search)")
    parser.add_argument("--skip-dtw", action="store_true",
                        help=argparse.SUPPRESS)  # internal flag; not in --help
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        raise FileNotFoundError(f"baseline not found: {args.baseline}")
    raw = json.loads(args.baseline.read_text())
    baseline = Baseline(
        primary=float(raw["primary"]),
        guards=GuardSet(**{k: float(v) for k, v in raw["guards"].items()}),
    )

    if not args.manifest.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"manifest at {args.manifest} is empty or not a list")

    corpus_root = args.corpus_root or args.manifest.parent.parent
    pseudo_truth_root = corpus_root / "pseudo_truth"
    score_by_piece = dict(DEFAULT_SCORE_BY_PIECE)
    corpus_scores = corpus_root / "scores" / "bach.prelude.bwv_846.json"
    if corpus_scores.exists():
        score_by_piece["bach_prelude_c_wtc1"] = corpus_scores

    config_body = json.loads(args.config.read_text())
    checkpoint_hash = config_body["checkpoint_hash"]
    parangonar_version = config_body["parangonar_version"]

    # Manifest n-pieces WARNING (stderr, not error).
    unique_pieces = {e["piece"] for e in manifest}
    if len(unique_pieces) < 2:
        print(
            f"WARNING: smoke-only baseline (n={len(unique_pieces)} piece(s)); "
            f"/autoresearch dispatch deferred until >=2 pieces have scores",
            file=sys.stderr,
        )

    results = _build_results(
        manifest,
        corpus_root=corpus_root,
        pseudo_truth_root=pseudo_truth_root,
        score_by_piece=score_by_piece,
        checkpoint_hash=checkpoint_hash,
        parangonar_version=parangonar_version,
        decim_hz_score_chroma=args.decim_hz,
        skip_dtw=args.skip_dtw,
        band_back_s=args.band_back_s,
        band_fwd_s=args.band_fwd_s,
        use_band=args.band_fwd_s >= 0,
    )

    metrics = aggregate(results, baseline=baseline, tolerance_s=args.tolerance_s)

    errors = [abs(r.error_seconds) for r in results if r.error_seconds is not None]
    if errors:
        errors_sorted = sorted(errors)
        n = len(errors_sorted)
        dist = {
            "mean": statistics.fmean(errors_sorted),
            "p50": errors_sorted[n // 2],
            "p90": errors_sorted[min(n - 1, int(0.9 * n))],
            "max": max(errors_sorted),
            "list": errors_sorted,
        }
        tolerance_sensitivity = {
            f"{tol}": (
                100.0 * sum(1 for e in errors_sorted if e <= tol) / n
            )
            for tol in TOLERANCE_SWEEP
        }
    else:
        dist = {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "list": []}
        tolerance_sensitivity = {f"{tol}": 0.0 for tol in TOLERANCE_SWEEP}

    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps({
        "primary": metrics.primary,
        "guards": metrics.guards.__dict__,
        "baseline": {
            "primary": baseline.primary,
            "guards": baseline.guards.__dict__,
        },
        "regressed": metrics.regressed,
        "n_chunks": len(results),
        "g2_threshold_scale": metrics.g2_threshold_scale,
        "error_seconds_distribution": dist,
        "tolerance_sensitivity": tolerance_sensitivity,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }, indent=2))
    print(f"{metrics.primary:.4f}")
    return 1 if metrics.regressed else 0


if __name__ == "__main__":
    sys.exit(main())
