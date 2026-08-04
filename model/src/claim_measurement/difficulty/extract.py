"""Backbone-agnostic per-piece extraction: iterate manifest entries, call the
backbone, write the shared .npz contract. Failures are recorded loudly, never
silently dropped (a bad MIDI or an OOM on one piece must not corrupt the run
or vanish from the report)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz
from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
from claim_measurement.difficulty.backbone import Backbone


@dataclass
class ExtractionReport:
    ok: int = 0
    skipped: int = 0
    failed: list = field(default_factory=list)


def _composer_id(composer: str, index_path: Path) -> int:
    """Look up (or append) composer in the shared composer_index.json.

    Ids are stable across backbones only when extraction runs are serialized
    against this index (the intended usage: run one backbone's extraction to
    completion, then the next). The read-modify-write below is not atomic or
    locked, so concurrent extraction runs racing on the same
    composer_index.json can assign divergent ids to the same composer."""
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = []
    if composer not in index:
        index.append(composer)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index))
    return index.index(composer)


def extract_embeddings(backbone: Backbone, entries: list[ManifestEntry],
                        midi_dir: Path, out_dir: Path, composer_index_path: Path,
                        skip_existing: bool = True) -> ExtractionReport:
    """Extract one .npz per entry. With skip_existing (the default), an entry
    whose .npz is already on disk is counted as skipped and the backbone is
    never called, so an interrupted GPU run resumes on the remainder instead of
    re-extracting everything. Safe only because write_embedding_npz is atomic:
    a present .npz is a complete .npz."""
    report = ExtractionReport()
    for entry in entries:
        out_path = Path(out_dir) / f"{entry.seg_id}.npz"
        if skip_existing and out_path.exists():
            report.skipped += 1
            continue
        midi_path = Path(midi_dir) / f"{entry.seg_id}.mid"
        try:
            embeddings = backbone.embed(midi_path)
            composer_id = _composer_id(entry.composer, composer_index_path)
            write_embedding_npz(out_path, embeddings,
                                 grade=entry.grade, composer_id=composer_id)
            report.ok += 1
        except Exception as exc:  # noqa: BLE001 -- record and continue; the run report is the source of truth
            report.failed.append(f"{entry.seg_id}: {exc!r}")
    return report
