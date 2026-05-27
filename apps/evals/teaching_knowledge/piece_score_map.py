from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORES_DIR = REPO_ROOT / "model" / "data" / "scores"

# Hand-curated mapping. Each entry MUST be verified by `ls model/data/scores/`
# before being added. Pieces not in this table return None.
#
# Verified 2026-05-27 against ls model/data/scores/:
#   bach_prelude_c_wtc1   -> bach.prelude.bwv_846.json (BWV 846 = WTC1 Prelude in C)
#   chopin_ballade_1      -> chopin.ballades.1.json
#   pathetique_mvt2       -> beethoven.piano_sonatas.8-2.json (Op.13 = Sonata 8, mvt 2)
#   chopin_etude_op10no4  -> chopin.etudes_op_10.4.json
#
# Skill_eval slugs intentionally NOT mapped (no matching score file in scores/
# directory as of 2026-05-27): moonlight_sonata_mvt1 (only 14-3 exists, not 14-1),
# fantaisie_impromptu, fur_elise, rachmaninoff_prelude_csm (ambiguous),
# chopin_waltz_csm, liszt_liebestraum_3, nocturne_op9no2, debussy_arabesque_1,
# bach_invention_1, clair_de_lune, mozart_k545_mvt1, schumann_traumerei,
# ensemble_4fold (meta).
PIECE_SCORE_MAP: dict[str, str] = {
    "bach_prelude_c_wtc1": "bach.prelude.bwv_846.json",
    "chopin_ballade_1": "chopin.ballades.1.json",
    "pathetique_mvt2": "beethoven.piano_sonatas.8-2.json",
    "chopin_etude_op10no4": "chopin.etudes_op_10.4.json",
}


def get_score_path_for_piece(piece_slug: str) -> Path | None:
    filename = PIECE_SCORE_MAP.get(piece_slug)
    if filename is None:
        return None
    path = SCORES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"piece_score_map: {piece_slug} -> {filename} does not exist under {SCORES_DIR}"
        )
    return path
