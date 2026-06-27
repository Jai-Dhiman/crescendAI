"""Dynamics G-A stage 2: non-degeneracy controls using the REAL production path.

Loads stage-1 per-(clip,scale) AMT velocity lists, builds bundles, runs the actual
DynamicsMeasurer + frozen route_verdict, and computes:
  (i)  performance-flip rate  -- verdict tracks the construction-known manipulation; PASS >= 0.80
  (ii) polarity-shuffle       -- |rate-0.5| collapses when polarities are randomized; PASS shift >= 0.20
plus a monotonicity check (d_soft < d_neutral < d_loud) on the signed statistic.

Run in the apps/evals env:
    cd apps/evals && uv run --extra all python \
        ../../model/src/claim_measurement/ga_validation/amt_dynamics_ga_metrics.py [data.json]
"""
import json
import random
import sys
from pathlib import Path

from claim_taxonomy.verdict_dispatch import route_verdict
from claim_taxonomy.verifier.measurers.dynamics import DynamicsMeasurer
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine
import claim_taxonomy

REPO = Path(__file__).resolve().parents[4]
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO / "model/data/results/ga_dynamics_velocities.json")
DATA = json.loads(DATA_PATH.read_text())
TAX = json.loads((Path(claim_taxonomy.__file__).resolve().parent / "claim_taxonomy.json").read_text())
REGISTRY = TAX["dimensions"]
TAU = float(REGISTRY["dynamics"]["tolerance"]["provisional"])
DUMMY_REGION = ResolvedRegion(audio_start_sec=0.0, audio_end_sec=1.0, alignment_uncertainty_sec=0.05, location_span_bars=5.0)


def bundle_from(vels):
    return {"notes": [{"onset": 0.1 * i, "offset": 0.1 * i + 0.05, "pitch": 60, "velocity": int(v)}
                      for i, v in enumerate(vels)]}


def measure_d(vels):
    if len(vels) < 20:
        return None
    return DynamicsMeasurer().measure("whole_piece", bundle_from(vels), DUMMY_REGION, SubstrateErrorEngine(seed=42))


def verdict(m, polarity):
    claim = {"dimension": "dynamics", "polarity": polarity,
             "_measurement": {"d": m.d, "tau": TAU, "error_bar": m.error_bar,
                              "event_count": m.event_count, "localizable": True, "substrate_failure": False}}
    return route_verdict(claim, REGISTRY)[0]


rows = {}
for clip, scales in DATA.items():
    ms = {s: measure_d(v) for s, v in scales.items()}
    if all(ms[s] is not None for s in ("soft", "neutral", "loud")):
        rows[clip] = ms
print(f"usable clips: {len(rows)} / {len(DATA)}  (tau={TAU})\n")

mono = sum(1 for m in rows.values() if m["soft"].d < m["neutral"].d < m["loud"].d)
direction = sum(1 for m in rows.values() if m["soft"].d < m["loud"].d)
print(f"[monotonic] d_soft<d_neutral<d_loud : {mono}/{len(rows)} = {mono/max(len(rows),1):.2f}")
print(f"[direction] d_soft<d_loud           : {direction}/{len(rows)} = {direction/max(len(rows),1):.2f}")
print(f"  median d  soft/neutral/loud = "
      f"{sorted(m['soft'].d for m in rows.values())[len(rows)//2]:+.1f} / "
      f"{sorted(m['neutral'].d for m in rows.values())[len(rows)//2]:+.1f} / "
      f"{sorted(m['loud'].d for m in rows.values())[len(rows)//2]:+.1f}")

flip_plus = sum(1 for m in rows.values()
                if verdict(m["loud"], "+") == "SUPPORTED" and verdict(m["soft"], "+") != "SUPPORTED")
flip_minus = sum(1 for m in rows.values()
                 if verdict(m["soft"], "-") == "SUPPORTED" and verdict(m["loud"], "-") != "SUPPORTED")
flip = (flip_plus + flip_minus) / (2 * max(len(rows), 1))
print(f"\n[flip +] loud SUPPORTED & soft not (claim '+'): {flip_plus}/{len(rows)}")
print(f"[flip -] soft SUPPORTED & loud not (claim '-'): {flip_minus}/{len(rows)}")
print(f"[performance-flip rate] = {flip:.2f}   (PASS >= 0.80)")

aligned = []
for m in rows.values():
    aligned.append(verdict(m["loud"], "+") == "SUPPORTED")
    aligned.append(verdict(m["soft"], "-") == "SUPPORTED")
rate_aligned = sum(aligned) / max(len(aligned), 1)
rng = random.Random(7)
shuf = []
for m in rows.values():
    for s in ("loud", "soft"):
        shuf.append(verdict(m[s], rng.choice(["+", "-"])) == "SUPPORTED")
rate_shuf = sum(shuf) / max(len(shuf), 1)
collapse = abs(rate_aligned - 0.5) - abs(rate_shuf - 0.5)
print(f"\n[polarity-shuffle] rate_aligned={rate_aligned:.2f} (|.-0.5|={abs(rate_aligned-0.5):.2f})  "
      f"rate_shuffled={rate_shuf:.2f} (|.-0.5|={abs(rate_shuf-0.5):.2f})")
print(f"[shuffle-collapse] |rate-0.5| shift = {collapse:.2f}   (PASS >= 0.20)")

print("\n=== DYNAMICS G-A VERDICT ===")
print(f"  performance-flip : {'PASS' if flip >= 0.80 else 'FAIL'} ({flip:.2f})")
print(f"  polarity-shuffle : {'PASS' if collapse >= 0.20 else 'FAIL'} ({collapse:.2f})")
