"""FRONT 7a-bis: build the two-arm teacher inputs for the voice-vs-input supply test.

7a found ZERO directional (rush/drag) timing supply in the generator corpus. Two
hypotheses for the NO_GO:
  H1 (voice): the warm "celebrate-strengths" prompt suppresses directional claims.
  H2 (input): the synthesizer's timing input is a single scalar QUALITY score
     (muq_means["timing"] ~0.57) with NO direction, so no grounded rush/drag claim is
     even possible.

This builds two paired input sets over the SAME 17 performances that have real AMT
onset bundles (front-5 gd_bundles), so the arms differ by exactly one variable:
  - ARM A: muq_means scalars only (the ORIGINAL production input).
  - ARM B: muq_means + a CONSTRUCTION-KNOWN directional tempo cue (a clean, well-formed
    "rushes/drags/steady" stimulus, some bar-localized), assigned deterministically.

Why construction-known and NOT a measured self-relative signal: raw AMT onsets are
polyphonic, so median(60/IOI) is texture-confounded -- chord notes have IOI~=0 and blow
the BPM up (Chopin Ballade "established ~375 BPM", window drifts -47..-75% = density
artifacts, not tempo). A trustworthy directional signal needs beat-tracking or score
alignment -- which is 7b itself. So ARM B cannot use a cheap real signal; it uses a
construction-known stimulus to isolate the PLUMBING question (like G-A injects known
corruption): given a CLEAN direction in its input, does the voice+format emit a
directional CLAIM? This is a capability probe, NOT a faithfulness label.

Interpretation:
  - ARM A directional > 0            -> H1 (voice): model confabulates direction from a
                                        scalar; supply is prompt-fixable but ungrounded.
  - ARM A ~0 but ARM B directional>0 -> H2 (input): model is honest (won't invent
                                        direction) but CAN express it when given one; the
                                        production blocker is the input lacks direction ->
                                        feed 7b's measurer output to the teacher. REVIVES 7.
  - both ~0                          -> H3 (fundamental voice/format) -> fork (a).

Run:
    uv run python model/src/claim_measurement/timing_supply/build_teacher_inputs.py \
        --bundles model/data/evals/gd_bundles/chopin_ballade_1 \
        --baseline apps/evals/results/baseline_v1.jsonl \
        --out-a /ABS/scratchpad/arm_a_inputs.json \
        --out-b /ABS/scratchpad/arm_b_inputs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[4]

# Construction-known directional stimuli (deterministic, index-assigned). A capability
# probe: does the voice+format convert a CLEAN direction into a directional CLAIM? The
# magnitudes/locations are plausible but INVENTED -- no faithfulness is claimed. The mix
# is directional-heavy (rush/drag) so the arm can actually exercise directional supply;
# 3 steady + 4 bar-localized also test the neutral and localization paths.
_STIMULI: list[dict] = [
    {"direction_label": "rush", "magnitude_pct": 12, "location": "whole_piece"},
    {"direction_label": "drag", "magnitude_pct": 10, "location": "whole_piece"},
    {"direction_label": "rush", "magnitude_pct": 18, "location": {"bar_start": 33, "bar_end": 40}},
    {"direction_label": "drag", "magnitude_pct": 9,  "location": "whole_piece"},
    {"direction_label": "rush", "magnitude_pct": 8,  "location": "whole_piece"},
    {"direction_label": "steady", "magnitude_pct": 2, "location": "whole_piece"},
    {"direction_label": "drag", "magnitude_pct": 14, "location": {"bar_start": 7, "bar_end": 12}},
    {"direction_label": "rush", "magnitude_pct": 11, "location": "whole_piece"},
    {"direction_label": "drag", "magnitude_pct": 7,  "location": "whole_piece"},
    {"direction_label": "rush", "magnitude_pct": 15, "location": {"bar_start": 90, "bar_end": 96}},
    {"direction_label": "steady", "magnitude_pct": 3, "location": "whole_piece"},
    {"direction_label": "drag", "magnitude_pct": 11, "location": "whole_piece"},
    {"direction_label": "rush", "magnitude_pct": 9,  "location": "whole_piece"},
    {"direction_label": "drag", "magnitude_pct": 16, "location": {"bar_start": 50, "bar_end": 58}},
    {"direction_label": "rush", "magnitude_pct": 13, "location": "whole_piece"},
    {"direction_label": "steady", "magnitude_pct": 1, "location": "whole_piece"},
    {"direction_label": "drag", "magnitude_pct": 10, "location": "whole_piece"},
]


def _stimulus(index: int) -> dict:
    s = dict(_STIMULI[index % len(_STIMULI)])
    s["construction_known"] = True
    s["caveat"] = "INVENTED directional stimulus (capability probe); not a measurement."
    return s


def _loc_phrase(loc) -> str:
    if isinstance(loc, dict):
        return f"in bars {loc['bar_start']}-{loc['bar_end']}"
    return "across the piece"


def _cue_text(sig: dict) -> str:
    """One natural-language directional cue the ARM-B teacher may turn into a claim."""
    lbl = sig["direction_label"]
    mag = sig["magnitude_pct"]
    where = _loc_phrase(sig["location"])
    head = ("Timing analysis vs the score (directional): ")
    if lbl == "rush":
        return (f"{head}the performance RUSHES {where}, running about {mag}% AHEAD of the "
                f"notated pulse (onsets systematically early).")
    if lbl == "drag":
        return (f"{head}the performance DRAGS {where}, running about {mag}% BEHIND the "
                f"notated pulse (onsets systematically late).")
    return (f"{head}the pace tracks the notated pulse {where} (within ~{mag}%, no "
            f"systematic rush or drag).")


def _muq_by_id(baseline: Path, piece: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in baseline.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("piece_slug") != piece:
            continue
        rid = r["recording_id"]
        if rid not in out and r.get("muq_means"):
            out[rid] = r["muq_means"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="timing_supply.build_teacher_inputs")
    ap.add_argument("--bundles", type=Path,
                    default=REPO / "model/data/evals/gd_bundles/chopin_ballade_1")
    ap.add_argument("--baseline", type=Path,
                    default=REPO / "apps/evals/results/baseline_v1.jsonl")
    ap.add_argument("--piece", default="chopin_ballade_1")
    ap.add_argument("--out-a", type=Path, required=True)
    ap.add_argument("--out-b", type=Path, required=True)
    args = ap.parse_args(argv)

    muq = _muq_by_id(args.baseline, args.piece)
    arm_a, arm_b = [], []
    index = 0
    for p in sorted(args.bundles.glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        b = json.loads(p.read_text())
        rid = b.get("video_id")
        means = muq.get(rid)
        if means is None:
            print(f"  skip {rid}: no muq_means in baseline", flush=True)
            continue
        sig = _stimulus(index)
        index += 1
        base = {"recording_id": rid, "piece": args.piece, "muq_means": means}
        arm_a.append(dict(base))
        arm_b.append({**base, "timing_direction_signal": sig,
                      "timing_direction_cue": _cue_text(sig)})

    for out, arm in ((args.out_a, arm_a), (args.out_b, arm_b)):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(arm, indent=2))

    labels = [x["timing_direction_signal"]["direction_label"] for x in arm_b]
    from collections import Counter
    print(f"ARM A: {len(arm_a)} inputs (scalar muq only) -> {args.out_a}", flush=True)
    print(f"ARM B: {len(arm_b)} inputs (+directional cue) -> {args.out_b}", flush=True)
    print(f"direction labels: {dict(Counter(labels))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
