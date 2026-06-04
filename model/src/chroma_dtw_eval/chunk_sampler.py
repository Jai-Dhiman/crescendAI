"""Practice-corpus chunk sampler.

Enumerates approved video_ids per piece (from candidates.yaml), cross-
references pseudo-truth coverage, stratifies positions across five
position buckets, and emits ChunkManifestEntry with PER-CHUNK
audio_sha256 computed from the chunk's audio bytes.

The manifest is committed under model/data/evals/chroma_dtw_fixtures/manifest.json
so verify.py and downstream baselines see the same chunks on every run.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import yaml

from chroma_dtw_eval.pseudo_truth_cache import cache_path

BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("intro", 0.00, 0.10),
    ("early", 0.10, 0.35),
    ("middle", 0.35, 0.65),
    ("late", 0.65, 0.90),
    ("cadence", 0.90, 1.00),
)
DEFAULT_CHUNK_LEN_S = 15.0
DEFAULT_N_PER_PIECE = 10


class PseudoTruthCoverageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkManifestEntry:
    piece: str
    video_id: str
    start_audio_sec: float
    end_audio_sec: float
    audio_sha256: str
    position_bucket: str


def _chunk_sha256(audio_path: Path, start_sec: float, end_sec: float) -> str:
    y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    start_f = max(0, int(round(start_sec * sr)))
    end_f = min(len(y), int(round(end_sec * sr)))
    h = hashlib.sha256(y[start_f:end_f].tobytes())
    return h.hexdigest()[:16]


def sample_chunks(
    corpus_root: Path,
    pseudo_truth_root: Path,
    seed: int,
    *,
    n_per_piece: int = DEFAULT_N_PER_PIECE,
    chunk_len_s: float = DEFAULT_CHUNK_LEN_S,
    manifest_out: Path | None = None,
) -> list[ChunkManifestEntry]:
    if n_per_piece < len(BUCKETS):
        raise ValueError(f"n_per_piece={n_per_piece} < {len(BUCKETS)} buckets")
    rng = random.Random(seed)
    per_bucket_base = n_per_piece // len(BUCKETS)
    remainder = n_per_piece - per_bucket_base * len(BUCKETS)
    counts = [per_bucket_base + (1 if i < remainder else 0) for i in range(len(BUCKETS))]

    practice_root = corpus_root / "practice_eval"
    entries: list[ChunkManifestEntry] = []
    pieces = sorted(p.name for p in practice_root.iterdir() if p.is_dir())
    for piece_id in pieces:
        yaml_path = practice_root / piece_id / "candidates.yaml"
        if not yaml_path.exists():
            continue
        body = yaml.safe_load(yaml_path.read_text()) or {}
        approved = [
            r for r in (body.get("recordings") or [])
            if r.get("approved") is True
        ]
        covered: list[tuple[str, float, Path]] = []
        for r in approved:
            vid = r["video_id"]
            pt_path = cache_path(pseudo_truth_root, piece_id, vid)
            if not pt_path.exists():
                continue
            data = json.loads(pt_path.read_text())
            perf = data.get("perf_audio_sec") or []
            if len(perf) < 2:
                continue
            duration_s = float(perf[-1])
            audio_path = practice_root / piece_id / "audio" / f"{vid}.wav"
            if not audio_path.exists() or duration_s <= chunk_len_s:
                continue
            covered.append((vid, duration_s, audio_path))
        if not covered:
            raise PseudoTruthCoverageError(
                f"no pseudo-truth coverage for piece {piece_id} "
                f"(checked {len(approved)} approved clips)"
            )
        for (name, lo, hi), count in zip(BUCKETS, counts):
            for _ in range(count):
                vid, dur, audio_path = covered[rng.randrange(len(covered))]
                lo_s = lo * dur
                hi_s = max(lo_s + 1e-3, hi * dur - chunk_len_s)
                start = rng.uniform(lo_s, hi_s)
                end = start + chunk_len_s
                sha = _chunk_sha256(audio_path, start, end)
                entries.append(ChunkManifestEntry(
                    piece=piece_id, video_id=vid,
                    start_audio_sec=start, end_audio_sec=end,
                    audio_sha256=sha, position_bucket=name,
                ))

    if manifest_out is not None:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps([asdict(e) for e in entries], indent=2))
    return entries
