"""The 37 hand features as a bake-off "backbone", so the #137 feature baseline
can be scored through the SAME folds as the frozen encoder arms (#138 Phase 0).

Why this exists: 0.8257 (MoonBeam, RidgeCV + seeded composer folds, bakeoff_cv.py)
is not comparable to 0.7929 (BASE_37, LightGBM + GroupKFold, tk_ablation.py).
Two different models over two different fold constructions -- the #135
cross-protocol mirage. The only honest reference for "does the encoder beat the
hand features?" is the hand features run through bakeoff_cv.py's own folds, which
is what this module produces.

Feature VALUES are read from tk_ablation.py's cache rather than recomputed, so
they are byte-identical to the ones #137 measured. The cache is guarded by the
same extractor fingerprint tk_ablation._load_matrix checks: if the source of
candidate_features/transkun_features has changed since the cache was built, this
raises instead of scoring stale values under new code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

N_BASE_FEATURES = 37
_FINGERPRINT_SOURCES = ("transkun_features.py", "phase3c_explore.py")


def extractor_fingerprint() -> str:
    """SHA of the modules that define the cached feature values. Same
    convention as tk_ablation._extractor_fingerprint (both files, in this
    order), so the digest is comparable to the one stored in the cache."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in _FINGERPRINT_SOURCES:
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


def load_feature37_cache(cache_path: Path,
                         expected_sha: str | None = None) -> dict[str, np.ndarray]:
    """{piece key -> 37-vector} from tk_ablation.py's feature cache.

    Feature ORDER is the insertion order of the first row's non-tk_ columns --
    identical to tk_ablation._load_matrix's `base` list, so the matrix here is
    the same matrix arm BASE_37 was scored on.

    expected_sha defaults to the live extractor fingerprint; pass it explicitly
    only in tests. A mismatch raises: silently scoring cached values that the
    current extractor would no longer produce is exactly the failure this guard
    exists to make loud.
    """
    cached = json.loads(Path(cache_path).read_text())
    rows = cached.get("rows") or []
    if not rows:
        raise ValueError(f"no rows in {cache_path}")
    want = extractor_fingerprint() if expected_sha is None else expected_sha
    if cached.get("extractor_sha") != want:
        raise ValueError(
            f"STALE FEATURE CACHE: {cache_path} was built by extractor "
            f"{cached.get('extractor_sha')!r} but the code on disk is {want!r}. "
            f"Re-run tk_ablation.py --stage extract before scoring.")
    names = [k for k in rows[0]
             if k not in ("key", "grade", "composer") and not k.startswith("tk_")]
    if len(names) != N_BASE_FEATURES:
        raise ValueError(f"expected {N_BASE_FEATURES} base features, "
                         f"cache has {len(names)}")
    return {r["key"]: np.array([r[n] for n in names], dtype=np.float32) for r in rows}


class CachedFeature37Backbone:
    """Backbone-shaped adapter over the cached feature matrix.

    Conforms to the Backbone protocol so extract_embeddings handles the composer
    index and the atomic .npz write exactly as it does for Aria and MoonBeam --
    the feature arm lands in emb/features37/ and `--stage eval` scores it with no
    special-casing.

    embed() is keyed by the MIDI stem (the manifest seg_id) via seg_id_to_key,
    because the cache is keyed by the PSyllabus piece key, not the seg_id.
    """

    def __init__(self, by_key: dict[str, np.ndarray], seg_id_to_key: dict[str, str],
                 pooling_name: str = "raw37"):
        self.by_key = by_key
        self.seg_id_to_key = seg_id_to_key
        self.pooling_name = pooling_name

    def embed(self, midi_path: Path) -> dict:
        seg_id = Path(midi_path).stem
        key = self.seg_id_to_key[seg_id]   # KeyError is the correct loud failure
        return {self.pooling_name: self.by_key[key]}
