"""Pedal G-A stage 2: non-degeneracy controls using the REAL production path (#101 front-3).

Loads the stage-1 render JSON, builds bundles from the AMT pedal events, runs the actual
PedalingMeasurer + frozen route_verdict, and computes:
  (i)  performance-flip rate -- verdict tracks the construction-known manipulation; PASS >= 0.80
  (ii) polarity-shuffle      -- |rate-0.5| collapses when polarities are randomized; PASS shift >= 0.20
plus a monotonicity check (d_sparse < d_neutral < d_dense) on the signed statistic.

Levels: sparse = less pedal ('-' under-pedaled), dense = more pedal ('+' over-pedaled).
Also reports the AMT-neutral on-fraction median -- the calibration source for
PedalingMeasurer.REFERENCE_FRACTION.

Run in the apps/evals env:
    cd apps/evals && uv run --extra all python \
        ../../model/src/claim_measurement/ga_validation/amt_pedaling_ga_metrics.py [data.json]
"""
import json
import random
import sys
from pathlib import Path

from claim_taxonomy.verdict_dispatch import route_verdict
from claim_taxonomy.verifier.measurers.pedaling import PedalingMeasurer, REFERENCE_FRACTION
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine
import claim_taxonomy

REPO = Path(__file__).resolve().parents[4]
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO / "model/data/results/ga_pedal_onfraction.json")
DATA = json.loads(DATA_PATH.read_text())
TAX = json.loads((Path(claim_taxonomy.__file__).resolve().parent / "claim_taxonomy.json").read_text())
REGISTRY = TAX["dimensions"]
TAU = float(REGISTRY["pedaling"]["tolerance"]["provisional"])
DUMMY_REGION = ResolvedRegion(audio_start_sec=0.0, audio_end_sec=1.0, alignment_uncertainty_sec=0.05, location_span_bars=5.0)


def bundle_from(total_dur, pedal_events):
    return {"notes": [{"onset": 0.0, "offset": float(total_dur), "pitch": 60, "velocity": 80}],
            "pedal_events": pedal_events}


def measure_d(total_dur, pedal_events):
    return PedalingMeasurer().measure("whole_piece", bundle_from(total_dur, pedal_events),
                                      DUMMY_REGION, SubstrateErrorEngine(seed=42))


def verdict(m, polarity):
    claim = {"dimension": "pedaling", "polarity": polarity,
             "_measurement": {"d": m.d, "tau": TAU, "error_bar": m.error_bar,
                              "event_count": m.event_count, "localizable": True, "substrate_failure": False}}
    return route_verdict(claim, REGISTRY)[0]


neutral_fracs = sorted(c["levels"]["neutral"]["amt_frac"] for c in DATA.values()
                       if c["levels"]["neutral"]["amt_frac"] is not None)
median_neutral = neutral_fracs[len(neutral_fracs) // 2] if neutral_fracs else float("nan")
print(f"AMT-neutral on-fraction: n={len(neutral_fracs)} median={median_neutral:.4f} "
      f"(baked REFERENCE_FRACTION={REFERENCE_FRACTION})\n")

rows = {}
for clip, c in DATA.items():
    td, lv = c["total_dur"], c["levels"]
    if not all(lv[k]["amt_frac"] is not None for k in ("sparse", "neutral", "dense")):
        continue
    rows[clip] = {level: measure_d(td, lv[level]["amt_pedal_events"]) for level in ("sparse", "neutral", "dense")}
print(f"usable clips: {len(rows)} / {len(DATA)}  (tau={TAU})\n")

mono = sum(1 for m in rows.values() if m["sparse"].d < m["neutral"].d < m["dense"].d)
direction = sum(1 for m in rows.values() if m["sparse"].d < m["dense"].d)
print(f"[monotonic] d_sparse<d_neutral<d_dense : {mono}/{len(rows)} = {mono/max(len(rows),1):.2f}")
print(f"[direction] d_sparse<d_dense           : {direction}/{len(rows)} = {direction/max(len(rows),1):.2f}")
print(f"  median d  sparse/neutral/dense = "
      f"{sorted(m['sparse'].d for m in rows.values())[len(rows)//2]:+.3f} / "
      f"{sorted(m['neutral'].d for m in rows.values())[len(rows)//2]:+.3f} / "
      f"{sorted(m['dense'].d for m in rows.values())[len(rows)//2]:+.3f}")

flip_plus = sum(1 for m in rows.values()
                if verdict(m["dense"], "+") == "SUPPORTED" and verdict(m["sparse"], "+") != "SUPPORTED")
flip_minus = sum(1 for m in rows.values()
                 if verdict(m["sparse"], "-") == "SUPPORTED" and verdict(m["dense"], "-") != "SUPPORTED")
flip = (flip_plus + flip_minus) / (2 * max(len(rows), 1))
print(f"\n[flip +] dense SUPPORTED & sparse not (claim '+'): {flip_plus}/{len(rows)}")
print(f"[flip -] sparse SUPPORTED & dense not (claim '-'): {flip_minus}/{len(rows)}")
print(f"[performance-flip rate] = {flip:.2f}   (PASS >= 0.80)")

aligned = []
for m in rows.values():
    aligned.append(verdict(m["dense"], "+") == "SUPPORTED")
    aligned.append(verdict(m["sparse"], "-") == "SUPPORTED")
rate_aligned = sum(aligned) / max(len(aligned), 1)
rng = random.Random(7)
shuf = []
for m in rows.values():
    for level in ("dense", "sparse"):
        shuf.append(verdict(m[level], rng.choice(["+", "-"])) == "SUPPORTED")
rate_shuf = sum(shuf) / max(len(shuf), 1)
collapse = abs(rate_aligned - 0.5) - abs(rate_shuf - 0.5)
print(f"\n[polarity-shuffle] rate_aligned={rate_aligned:.2f} (|.-0.5|={abs(rate_aligned-0.5):.2f})  "
      f"rate_shuffled={rate_shuf:.2f} (|.-0.5|={abs(rate_shuf-0.5):.2f})")
print(f"[shuffle-collapse] |rate-0.5| shift = {collapse:.2f}   (PASS >= 0.20)")

print("  (symmetric flip mixes the saturated over-pedal direction; see DIRECTIONAL below)")

# --- DIRECTIONAL G-A (the reported verdict): AMT pedal saturates, so over-pedal '+' is
# substrate-insensitive (scoped out, UNVERIFIABLE in production). The validated tier is
# UNDER-pedal '-'. Performance-flip + polarity-shuffle are computed on the '-' direction. ---
flip_under = sum(1 for m in rows.values()
                 if verdict(m["sparse"], "-") == "SUPPORTED" and verdict(m["dense"], "-") != "SUPPORTED")
flip_under_rate = flip_under / max(len(rows), 1)
true_supp = sum(1 for m in rows.values() if verdict(m["sparse"], "-") == "SUPPORTED") / max(len(rows), 1)
rng2 = random.Random(7)
shuf_u = [verdict(m["sparse"], rng2.choice(["+", "-"])) == "SUPPORTED" for m in rows.values()]
shuf_supp = sum(shuf_u) / max(len(shuf_u), 1)
collapse_u = abs(true_supp - 0.5) - abs(shuf_supp - 0.5)
over_dense_med = sorted(m["dense"].d for m in rows.values())[len(rows) // 2]
print("\n=== PEDAL G-A (DIRECTIONAL, under-pedal scope -- reported) ===")
print(f"  under-pedal performance-flip : {'PASS' if flip_under_rate >= 0.80 else 'FAIL'} "
      f"({flip_under_rate:.2f}; dry '-' SUPPORTED & wet '-' not: {flip_under}/{len(rows)})")
print(f"  under-pedal polarity-shuffle : {'PASS' if collapse_u >= 0.20 else 'FAIL'} "
      f"({collapse_u:+.2f}; aligned {true_supp:.2f} -> shuffled {shuf_supp:.2f})")
print(f"  over-pedal '+' : SCOPED OUT (substrate-insensitive); dense median d={over_dense_med:+.3f} "
      f"(full pedal, true frac=1.0, below tau -- AMT saturation)")
