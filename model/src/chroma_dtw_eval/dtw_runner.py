"""Subprocess wrapper over apps/api/src/wasm/score-analysis/target/release/dtw_chunk_cli.

Raises DtwRunnerError on any binary failure — no silent fallbacks. The binary
must be built ahead of time (e.g. by Task A5's smoke test or by the verify CLI
on first run); run_dtw shells out, sends chroma on stdin, parses JSON stdout.
"""
from __future__ import annotations

import json
import subprocess
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


def run_dtw(
    chroma: np.ndarray, score_bars_path: Path,
    frame_rate_hz: float, decim_hz: float,
) -> DtwResult:
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise DtwRunnerError(f"chroma must be (12, N), got {chroma.shape}")
    if chroma.dtype != np.float32:
        raise DtwRunnerError(f"chroma dtype must be float32, got {chroma.dtype}")
    if not score_bars_path.exists():
        raise DtwRunnerError(f"score bars file not found: {score_bars_path}")
    binary = _ensure_binary()
    n_audio = chroma.shape[1]
    res = subprocess.run(
        [str(binary), str(score_bars_path), str(frame_rate_hz), str(decim_hz), str(n_audio)],
        input=chroma.flatten().astype(np.float32).tobytes(),
        capture_output=True, timeout=30,
    )
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
