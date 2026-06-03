"""Catalog coverage and quality acceptance harness.

check_coverage verifies that every (slug -> piece_id) in a mapping resolves to
an existing, non-trivial, monotonic score JSON. It is TDD'd against tmp_path
fixtures and RUN against the real catalog as the feature's acceptance gate.
No chroma logic anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Canonical 16-entry slug -> piece_id map (the #21 chroma-harness contract).
CANONICAL_MAP: dict[str, str] = {
    "bach_invention_1": "bach.inventions.1",
    "bach_prelude_c_wtc1": "bach.prelude.bwv_846",
    "chopin_ballade_1": "chopin.ballades.1",
    "chopin_etude_op10no4": "chopin.etudes_op_10.4",
    "chopin_waltz_csm": "chopin.waltzes.64-2",
    "clair_de_lune": "debussy.suite_bergamasque.3_clair_de_lune",
    "debussy_arabesque_1": "debussy.deux_arabesques.1",
    "fantaisie_impromptu": "chopin.fantaisie_impromptu",
    "fur_elise": "beethoven.fur_elise",
    "liszt_liebestraum_3": "liszt.liebestraume.3",
    "moonlight_sonata_mvt1": "beethoven.piano_sonatas.14-1",
    "mozart_k545_mvt1": "mozart.piano_sonatas.16-1",
    "nocturne_op9no2": "chopin.nocturnes.9-2",
    "pathetique_mvt2": "beethoven.piano_sonatas.8-2",
    "rachmaninoff_prelude_csm": "rachmaninoff.preludes_op_3.2",
    "schumann_traumerei": "schumann.kinderszenen.7",
}


def check_coverage(scores_dir: Path, mapping: dict[str, str]) -> list[str]:
    """Verify each (slug, piece_id) resolves to a non-trivial, monotonic score.

    For each entry: if {piece_id}.json is missing, append a MISSING line.
    Otherwise load it, flatten all note onsets across bars, and append failure
    strings for < 20 notes, total_bars < 1, or non-monotonic onsets.

    Returns an empty list when every entry passes.
    """
    failures: list[str] = []
    for slug, piece_id in mapping.items():
        score_path = scores_dir / f"{piece_id}.json"
        if not score_path.exists():
            failures.append(f"{slug}: MISSING {piece_id}.json")
            continue

        with open(score_path) as f:
            data = json.load(f)

        onsets: list[float] = []
        for bar in data.get("bars", []):
            for note in bar.get("notes", []):
                onsets.append(note["onset_seconds"])

        if len(onsets) < 20:
            failures.append(f"{slug}: {piece_id} has {len(onsets)} notes (< 20 minimum)")
        if data.get("total_bars", 0) < 1:
            failures.append(f"{slug}: {piece_id} has total_bars < 1")
        if any(onsets[i] < onsets[i - 1] for i in range(1, len(onsets))):
            failures.append(f"{slug}: {piece_id} onsets are not monotonic")

    return failures
