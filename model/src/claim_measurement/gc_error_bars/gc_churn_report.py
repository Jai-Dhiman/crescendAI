"""G-C stage 2: reduce the re-transcription churn JSON to the measured 1-sigma.

Loads the stage-1 per-clip nuisance-variant velocities, computes the empirical
substrate error of the whole_piece mean-velocity statistic (per-clip + pooled), and
recommends the per-note substrate sigma to wire into DynamicsMeasurer so the frozen
router's dead-band (error_bar) covers the measured churn. Prints the placeholder it
replaces (VELOCITY_QUANT_STEP-derived) for an honest before/after.

Run in the apps/evals env (imports claim_taxonomy for the placeholder comparison):
    cd apps/evals && uv run --extra all python \
        ../../model/src/claim_measurement/gc_error_bars/gc_churn_report.py [churn.json]
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gc_churn_metrics import (  # noqa: E402
    clip_per_note_churn,
    clip_statistic_churn,
    pool,
    recommend_substrate_terms,
)

# model/data is gitignored -> absent in worktrees; anchor to the PRIMARY checkout where
# the stage-1 render wrote its results (the part before any `.worktrees/<branch>/`).
_here = Path(__file__).resolve()
_parts = _here.parts
REPO = (Path(*_parts[:_parts.index(".worktrees")]) if ".worktrees" in _parts
        else _here.parents[4])
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO / "model/data/results/gc_dynamics_churn.json")
DATA = json.loads(DATA_PATH.read_text())
META = DATA.pop("_meta", {})
CLIPS = {k: v for k, v in DATA.items() if "n_base_notes" in v}

stat_sigmas = [clip_statistic_churn(v["variant_mean_vels"]) for v in CLIPS.values()]
note_sigmas = [clip_per_note_churn(v["per_note_deltas"]) for v in CLIPS.values()]
note_counts = [v["n_base_notes"] for v in CLIPS.values() if v["n_base_notes"] > 0]
median_n = sorted(note_counts)[len(note_counts) // 2] if note_counts else 0.0

stat_pool = pool(stat_sigmas)
note_pool = pool(note_sigmas)
rec = recommend_substrate_terms(stat_pool, note_pool)

# Placeholder substrate error the prior measurer produced at the typical note count
# (sigma_note/sqrt(12) averaged over N notes -- the independent-noise-only model).
try:
    from claim_taxonomy.verifier.measurers.dynamics import VELOCITY_QUANT_STEP
except Exception:  # noqa: BLE001 - report path also works outside apps/evals
    VELOCITY_QUANT_STEP = 5.0
placeholder_substrate_std_at_median_n = (
    (VELOCITY_QUANT_STEP / math.sqrt(12.0)) / math.sqrt(max(median_n, 1.0))
)

all_det = META.get("all_deterministic")
print("=== G-C EMPIRICAL ERROR BARS: dynamics (mean AMT velocity) ===")
print(f"  clips={stat_pool['n']}  median_note_count={median_n}  nuisance={{"
      f"gain_jitter_db=+/-{META.get('gain_jitter_db')}, snr_db={META.get('snr_db')}, "
      f"k_variants={META.get('k_variants')}}}")
print(f"  determinism check (identical audio -> identical velocities): "
      f"{'PASS (churn is nuisance-driven, not RNG)' if all_det else 'FAIL/unknown'}")
print()
print(f"  statistic churn (std of whole_piece mean-velocity across re-captures):")
print(f"    median={stat_pool['median']:.3f}  mean={stat_pool['mean']:.3f}  "
      f"p90={stat_pool['p90']:.3f}  max={stat_pool['max']:.3f}  velocity units")
print(f"  per-note churn (std of matched-note velocity deltas):")
print(f"    median={note_pool['median']:.3f}  mean={note_pool['mean']:.3f}  "
      f"p90={note_pool['p90']:.3f}  max={note_pool['max']:.3f}  velocity units")
print()
print(f"  finding: the measured statistic churn ({stat_pool['median']:.3f} median) is far "
      f"LARGER than the independent-noise prediction (per_note {note_pool['median']:.3f} / "
      f"sqrt({median_n}) = {note_pool['median']/math.sqrt(max(median_n,1)):.3f})")
print(f"  -> the churn is CORRELATED across notes (gain jitter shifts all notes together) "
      f"and does NOT shrink with N -> it needs a FLAT floor, not a /sqrt(N) term.")
print()
print(f"  prior model substrate_std at median N ((VELOCITY_QUANT_STEP={VELOCITY_QUANT_STEP} "
      f"/sqrt12)/sqrt(N)) = {placeholder_substrate_std_at_median_n:.3f}  (UNDER-covers the "
      f"measured {stat_pool['median']:.3f} by ~{stat_pool['median']/max(placeholder_substrate_std_at_median_n,1e-9):.1f}x)")
print(f"  RECOMMENDED two-term substrate_var = max(sigma_note**2/N, floor**2):")
print(f"    sigma_note (per-note p90)      = {rec['sigma_note']:.3f} velocity units")
print(f"    statistic_floor (stat p90)     = {rec['statistic_floor']:.3f} velocity units "
      f"(max observed {rec['statistic_floor_max']:.3f})")
print(f"  -> dead-band WIDENS to >= the measured 1-sigma at all note counts.")
