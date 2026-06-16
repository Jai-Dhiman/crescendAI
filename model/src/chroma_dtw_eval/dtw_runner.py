"""Subprocess wrapper over apps/api/src/wasm/score-analysis/target/release/dtw_chunk_cli.

Raises DtwRunnerError on any binary failure — no silent fallbacks. The binary
must be built ahead of time (e.g. by Task A5's smoke test or by the verify CLI
on first run); run_dtw shells out, sends chroma on stdin, parses JSON stdout.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DtwRunnerError(RuntimeError):
    pass


@dataclass
class DtwResult:
    predicted_score_frame: int
    cost: float
    bar_min: int
    bar_max: int
    bar_per_frame: list[int]


_REPO_ROOT = Path(__file__).resolve().parents[3]
_BIN = _REPO_ROOT / "apps/api/src/wasm/score-analysis/target/release/dtw_chunk_cli"


def _ensure_binary() -> Path:
    if _BIN.exists():
        return _BIN
    crate = _REPO_ROOT / "apps/api/src/wasm/score-analysis"
    res = subprocess.run(
        ["cargo", "build", "--release", "--bin", "dtw_chunk_cli"],
        cwd=crate, capture_output=True, text=True,
    )
    if res.returncode != 0 or not _BIN.exists():
        raise DtwRunnerError(f"failed to build dtw_chunk_cli: {res.stderr}")
    return _BIN


def _resolve_bars_file(score_path: Path) -> Path:
    """Return a path to a JSON file holding a bare array of ScoreBar objects.

    Accepts either a bare array (passed through unchanged) or the full score
    object with a top-level "bars" key (extracted into a NamedTemporaryFile the
    caller must delete). Raises DtwRunnerError on any other shape.
    """
    body = json.loads(score_path.read_text())
    if isinstance(body, list):
        return score_path
    if isinstance(body, dict) and isinstance(body.get("bars"), list):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".bars.json", delete=False
        )
        try:
            json.dump(body["bars"], tmp)
        finally:
            tmp.close()
        return Path(tmp.name)
    raise DtwRunnerError(
        f"score JSON at {score_path} is neither a ScoreBar array nor an object "
        f"with a 'bars' array"
    )


def run_dtw(
    chroma: np.ndarray, score_bars_path: Path,
    frame_rate_hz: float, decim_hz: float,
    prior_score_frame: int = -1,
    band_back_frames: int = 0,
    band_fwd_frames: int = 0,
) -> DtwResult:
    """Run the chunk DTW. `prior_score_frame < 0` disables the local-margin band
    (global endpoint search); otherwise the endpoint is constrained to
    [prior - band_back_frames, prior + band_fwd_frames)."""
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise DtwRunnerError(f"chroma must be (12, N), got {chroma.shape}")
    if chroma.dtype != np.float32:
        raise DtwRunnerError(f"chroma dtype must be float32, got {chroma.dtype}")
    if not score_bars_path.exists():
        raise DtwRunnerError(f"score bars file not found: {score_bars_path}")
    binary = _ensure_binary()
    n_audio = chroma.shape[1]

    # The CLI deserializes a bare `Vec<ScoreBar>`. Score JSONs on disk are the
    # full score object ({source, piece_id, ..., bars:[...]}); extract the bars
    # array into a temp file so the CLI parses. A file that is already a JSON
    # array is passed through unchanged.
    bars_file = _resolve_bars_file(score_bars_path)
    try:
        res = subprocess.run(
            [str(binary), str(bars_file), str(frame_rate_hz), str(decim_hz), str(n_audio),
             str(int(prior_score_frame)), str(int(band_back_frames)), str(int(band_fwd_frames))],
            input=chroma.flatten().astype(np.float32).tobytes(),
            capture_output=True, timeout=30,
        )
    finally:
        if bars_file != score_bars_path:
            bars_file.unlink(missing_ok=True)
    if res.returncode != 0:
        raise DtwRunnerError(
            f"dtw_chunk_cli exited {res.returncode}: {res.stderr.decode('utf-8', 'replace')}"
        )
    parsed = json.loads(res.stdout.decode("utf-8"))
    return DtwResult(
        predicted_score_frame=int(parsed["predicted_score_frame"]),
        cost=float(parsed["cost"]),
        bar_min=int(parsed["bar_min"]),
        bar_max=int(parsed["bar_max"]),
        bar_per_frame=[int(b) for b in parsed["bar_per_frame"]],
    )
