"""Contrast-pair manifest: YAML schema v1 + loader for the Gate 0 probe.

Schema (YAML):
    schema_version: 1
    sample_rate: 16000            # every clip header must match
    pairs:
      - id: unique-string
        axis: pedaling            # pedaling | dynamics | phrasing
        population: real          # real | synthetic  (NEVER pooled downstream)
        clip_a: relative/path.wav # relative to repo_root (model/)
        clip_b: relative/path.wav
        degraded: a               # which clip exhibits the degradation on axis
        description: free text

Loading validates EVERY clip: existence (with the exact rehydrate command
in the error when the path is R2-offloaded per data/manifests/r2_offload.json)
and WAV header (mono, expected sample rate, not truncated). A manifest
that cannot fully load raises -- probing a silently filtered subsample
is impossible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from audio_teacher.audio import validate_wav

MODEL_ROOT = Path(__file__).resolve().parents[2]  # model/
DEFAULT_OFFLOAD_REGISTRY = MODEL_ROOT / "data" / "manifests" / "r2_offload.json"

AXES = ("pedaling", "dynamics", "phrasing")
POPULATIONS = ("real", "synthetic")
_REQUIRED_PAIR_KEYS = (
    "id", "axis", "population", "clip_a", "clip_b", "degraded", "description",
)


class ManifestError(Exception):
    """The manifest file itself violates the schema."""


@dataclass(frozen=True)
class ContrastPair:
    pair_id: str
    axis: str
    population: str
    clip_a: Path
    clip_b: Path
    degraded: str  # "a" | "b"
    description: str


@dataclass(frozen=True)
class ProbeManifest:
    sample_rate: int
    pairs: tuple[ContrastPair, ...]


def _ensure_clip_local(path: Path, repo_root: Path, offload_registry: Path) -> None:
    """Raise FileNotFoundError with the exact rehydrate command for a missing
    clip. Mirrors src/paths.py ensure_local (not importable from installed
    packages), extended with prefix matching so clip FILES under an offloaded
    DIRECTORY (e.g. data/raw/competition) resolve to that directory's entry.
    """
    if path.exists():
        return
    hint = ""
    if offload_registry.exists():
        with offload_registry.open() as f:
            registry = json.load(f)
        try:
            rel = str(path.relative_to(repo_root))
        except ValueError:
            rel = str(path)
        for registered, entry in registry.get("entries", {}).items():
            if rel == registered or rel.startswith(registered + "/"):
                if "r2_prefix" in entry:
                    cmd = (
                        f"rclone copy {registry['remote_name']}:{registry['bucket']}"
                        f"/{entry['r2_prefix']} {registered}"
                    )
                else:
                    cmd = entry.get("regen_command", "")
                hint = (
                    f" Clip is under R2-offloaded path ({entry.get('reason', '')})."
                    f" Rehydrate with:\n    {cmd}"
                )
                break
    raise FileNotFoundError(f"probe clip missing: {path}.{hint}")


def load_manifest(
    manifest_path: Path | str,
    repo_root: Path = MODEL_ROOT,
    offload_registry: Path = DEFAULT_OFFLOAD_REGISTRY,
) -> ProbeManifest:
    manifest_path = Path(manifest_path)
    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ManifestError(f"{manifest_path}: expected a mapping with schema_version: 1")
    sample_rate = raw.get("sample_rate")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ManifestError(f"{manifest_path}: sample_rate must be a positive integer")
    raw_pairs = raw.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ManifestError(f"{manifest_path}: pairs must be a non-empty list")

    pairs: list[ContrastPair] = []
    seen: set[str] = set()
    for i, rp in enumerate(raw_pairs):
        missing = [k for k in _REQUIRED_PAIR_KEYS if k not in rp]
        if missing:
            raise ManifestError(f"{manifest_path}: pair[{i}] missing keys {missing}")
        pair_id = rp["id"]
        if pair_id in seen:
            raise ManifestError(f"{manifest_path}: duplicate pair id {pair_id!r}")
        seen.add(pair_id)
        if rp["axis"] not in AXES:
            raise ManifestError(
                f"{manifest_path}: pair {pair_id!r} axis {rp['axis']!r} not in {AXES}"
            )
        if rp["population"] not in POPULATIONS:
            raise ManifestError(
                f"{manifest_path}: pair {pair_id!r} population {rp['population']!r} "
                f"not in {POPULATIONS}"
            )
        if rp["degraded"] not in ("a", "b"):
            raise ManifestError(
                f"{manifest_path}: pair {pair_id!r} degraded must be 'a' or 'b'"
            )
        clip_a = repo_root / rp["clip_a"]
        clip_b = repo_root / rp["clip_b"]
        for clip in (clip_a, clip_b):
            _ensure_clip_local(clip, repo_root, offload_registry)
            validate_wav(clip, expected_sample_rate=sample_rate)
        pairs.append(
            ContrastPair(
                pair_id=pair_id,
                axis=rp["axis"],
                population=rp["population"],
                clip_a=clip_a,
                clip_b=clip_b,
                degraded=rp["degraded"],
                description=rp["description"],
            )
        )
    return ProbeManifest(sample_rate=sample_rate, pairs=tuple(pairs))
