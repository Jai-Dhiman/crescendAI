"""Two-arm dynamics level-supply + sign-fidelity scorer (#101 / #67). NO LLM in the truth path.

The deterministic complement to route_and_score.py for the dynamics level-cue probe. Reads
the LLM-extracted dynamics claims for BOTH arms (schema = gd_rate/extract_prompt.md) plus the
ARM-B inputs (which carry the measured cue label per performance) and computes:

  - per arm: subtype histogram (level / contrast / ambiguous), whole-piece level-claim supply,
    distinct performances with >=1 whole-piece level claim, polarity histogram within level.
  - supply LIFT (the headline): ARM A level supply vs ARM B level supply, with the front-7
    interpretation (voice vs input vs fundamental block for dynamics-level).
  - sign fidelity (ARM B): for each ARM-B whole-piece level claim, the expected polarity from
    its cue label (loud->+, soft->-, balanced->neutral) vs the extracted polarity. This is the
    real measurement the signal-fidelity rate reduces to -- the teacher can dilute, invert, or
    neutralize a cue near the reference.

The faithfulness RATE itself (SUPPORTED/(SUPPORTED+REFUTED) via the real measurer + frozen
router) is produced by route_and_score.py; this script quantifies WHERE the claims come from
and whether their SIGN tracks the cue, which the rate alone cannot show.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[4]

IN_SCOPE_SUBTYPE = "level"
# below this many whole-piece level claims, treat supply as materially ~0 (NO lift).
MIN_LEVEL_SUPPLY = 5
EXPECTED_POLARITY = {"loud": "+", "soft": "-", "balanced": "neutral"}


def _is_whole_piece(loc) -> bool:
    return loc == "whole_piece" or not isinstance(loc, dict)


def level_claims(claims: list[dict]) -> list[dict]:
    """Whole-piece level claims -- the in-scope supply for the mean-velocity statistic."""
    return [c for c in claims
            if c.get("dynamics_subtype") == IN_SCOPE_SUBTYPE and _is_whole_piece(c.get("location"))]


def subtype_histogram(claims: list[dict]) -> dict[str, int]:
    return dict(Counter(c.get("dynamics_subtype", "ambiguous") for c in claims))


def arm_summary(claims: list[dict]) -> dict:
    lvl = level_claims(claims)
    return {
        "n_claims": len(claims),
        "subtype_histogram": subtype_histogram(claims),
        "n_level_whole_piece": len(lvl),
        "n_perfs_with_level": len({c["recording_id"] for c in lvl}),
        "level_polarity_histogram": dict(Counter(c.get("polarity") for c in lvl)),
    }


def expected_polarity(cue_label: str) -> str:
    return EXPECTED_POLARITY[cue_label]


def sign_fidelity(level_claims_b: list[dict], cue_by_rid: dict[str, str]) -> dict:
    """Fraction of ARM-B whole-piece level claims whose extracted polarity matches the
    polarity implied by that performance's measured cue label. Claims for a recording with
    no cue label are skipped (reported as n_no_cue)."""
    n = n_correct = n_no_cue = 0
    confusion: dict[str, int] = {}
    for c in level_claims_b:
        lbl = cue_by_rid.get(c["recording_id"])
        if lbl is None:
            n_no_cue += 1
            continue
        want = expected_polarity(lbl)
        got = c.get("polarity")
        n += 1
        if got == want:
            n_correct += 1
        key = f"{lbl}:{want}->{got}"
        confusion[key] = confusion.get(key, 0) + 1
    return {
        "n_scored": n,
        "n_correct": n_correct,
        "sign_fidelity": (n_correct / n) if n else None,
        "n_no_cue": n_no_cue,
        "confusion": confusion,
    }


def supply_interpretation(n_level_a: int, n_level_b: int, min_supply: int = MIN_LEVEL_SUPPLY) -> str:
    a_present = n_level_a >= min_supply
    b_present = n_level_b >= min_supply
    if a_present:
        return ("voice: the teacher states whole-piece loudness LEVEL even without a cue "
                "(ungrounded -- confabulated from a scalar).")
    if b_present:
        return ("input: ARM A ~0 but ARM B lifts level supply -- the teacher is honest (won't "
                "invent overall loudness) but expresses it when given a measured signal. The "
                "front-5 dynamics supply gap is PROMPTABLE via the input.")
    return ("fundamental: neither arm produces whole-piece level claims -- the voice/format "
            "resists overall-loudness LEVEL statements even when cued.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.classify_supply")
    ap.add_argument("--claims-a", type=Path, required=True)
    ap.add_argument("--claims-b", type=Path, required=True)
    ap.add_argument("--inputs-b", type=Path, required=True,
                    help="dyn_arm_b_inputs.json -- carries dynamics_level_signal.label per rid")
    ap.add_argument("--out", type=Path,
                    default=REPO / "model/data/results/dynamics_level_cue_supply.json")
    args = ap.parse_args(argv)

    claims_a = json.loads(args.claims_a.read_text())
    claims_b = json.loads(args.claims_b.read_text())
    inputs_b = json.loads(args.inputs_b.read_text())
    cue_by_rid = {x["recording_id"]: x["dynamics_level_signal"]["label"] for x in inputs_b}

    sum_a = arm_summary(claims_a)
    sum_b = arm_summary(claims_b)
    fidelity = sign_fidelity(level_claims(claims_b), cue_by_rid)
    interp = supply_interpretation(sum_a["n_level_whole_piece"], sum_b["n_level_whole_piece"])

    result = {
        "probe": "dynamics_level_cue_supply",
        "n_performances": len(inputs_b),
        "cue_label_distribution": dict(Counter(cue_by_rid.values())),
        "arm_a_no_cue": sum_a,
        "arm_b_with_cue": sum_b,
        "supply_lift": {
            "arm_a_level_whole_piece": sum_a["n_level_whole_piece"],
            "arm_b_level_whole_piece": sum_b["n_level_whole_piece"],
            "interpretation": interp,
        },
        "sign_fidelity_arm_b": fidelity,
        "caveat": ("The cue is a MEASURED self-relative signal (mean AMT velocity); the scorer "
                   "uses the SAME statistic, so the downstream rate is signal-fidelity, not an "
                   "independent faithfulness rate. Soft (-) direction is untested if no cue "
                   "labeled soft. Register-specific vs overall-loudness level is not separately "
                   "gated here (the cue is whole-piece-overall by construction)."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n=== dynamics level-cue supply (two-arm) ===", flush=True)
    print(f"perfs={len(inputs_b)}  cue labels={result['cue_label_distribution']}", flush=True)
    print(f"ARM A level@whole_piece = {sum_a['n_level_whole_piece']} "
          f"(subtypes {sum_a['subtype_histogram']})", flush=True)
    print(f"ARM B level@whole_piece = {sum_b['n_level_whole_piece']} "
          f"(subtypes {sum_b['subtype_histogram']})", flush=True)
    print(f"supply interpretation: {interp}", flush=True)
    print(f"ARM B sign-fidelity = {fidelity['sign_fidelity']} "
          f"({fidelity['n_correct']}/{fidelity['n_scored']})  confusion={fidelity['confusion']}",
          flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
