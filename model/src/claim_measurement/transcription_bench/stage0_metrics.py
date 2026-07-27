# /// script
# requires-python = ">=3.11"
# dependencies = ["mir_eval>=0.7","pretty_midi>=0.2.10","numpy>=1.24.0","scipy>=1.10.0"]
# ///
"""Stage 0 metrics: aria-amt vs Transkun (vs Kong) on EXPRESSIVE fidelity, not just note-F1.

For each MAESTRO clip, compare every transcriber's MIDI to the ground-truth MIDI on:
  * note onset F1            (mir_eval, offset-free)
  * note onset+offset F1     (mir_eval, offset_ratio=0.2)
  * note+offset+velocity F1  (mir_eval.transcription_velocity)
  * velocity MAE + Spearman  (on onset-matched notes; Spearman = does it capture RELATIVE dynamics,
                              the thing CrescendAI + #104 fragile features actually need)
  * onset timing MAE (ms)    (on matched notes)
  * sustain-pedal frame F1   (CC64 binarized at >=64 on a 100Hz grid)

The velocity/pedal/timing rows are the ones that matter here -- note-F1 is known to be inadequate for
expressive fidelity (JKU ISMIR 2024). Run: uv run --script stage0_metrics.py
"""
import json
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.stats import spearmanr
import mir_eval

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
BENCH = PRIMARY / "model/data/results/transcription_bench"
GT = BENCH / "gt"
OUT = PRIMARY / "model/data/results/mirex_stage0_transcription_bench.json"
MODEL_DIRS = {"aria_amt": BENCH / "aria_mid", "transkun": BENCH / "transkun_mid",
              "kong": BENCH / "kong_mid"}
ONSET_TOL = 0.05  # 50ms, mir_eval default
WINDOW = 30.0


def _notes(pm_path: Path):
    """(intervals[N,2], pitches_hz[N], velocities[N]) from a MIDI file's non-drum notes."""
    pm = pretty_midi.PrettyMIDI(str(pm_path))
    iv, pit, vel = [], [], []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            iv.append([n.start, max(n.end, n.start + 1e-3)])
            pit.append(n.pitch)
            vel.append(n.velocity)
    if not iv:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    order = np.argsort([a[0] for a in iv])
    iv = np.array(iv)[order]
    pit = np.array(pit)[order]
    vel = np.array(vel)[order]
    return iv, pit, vel


def _pedal_frames(pm_path: Path, hz: int = 100):
    """Binary sustain-pedal (CC64>=64) state on a fixed grid over [0, WINDOW]."""
    pm = pretty_midi.PrettyMIDI(str(pm_path))
    grid = np.zeros(int(WINDOW * hz), dtype=bool)
    events = sorted((c.time, c.value) for inst in pm.instruments
                    for c in inst.control_changes if c.number == 64)
    state, ei = 0, 0
    for f in range(len(grid)):
        t = f / hz
        while ei < len(events) and events[ei][0] <= t:
            state = events[ei][1]
            ei += 1
        grid[f] = state >= 64
    return grid


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(ref_path: Path, est_path: Path) -> dict:
    ri, rp, rv = _notes(ref_path)
    ei, ep, ev = _notes(est_path)
    rp_hz = mir_eval.util.midi_to_hz(rp)
    ep_hz = mir_eval.util.midi_to_hz(ep)

    def prf(offset_ratio):
        if len(ri) == 0 or len(ei) == 0:
            return 0.0
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ri, rp_hz, ei, ep_hz, onset_tolerance=ONSET_TOL, offset_ratio=offset_ratio)
        return float(f)

    onset_f1 = prf(None)
    onoff_f1 = prf(0.2)

    # velocity F1 (mir_eval.transcription_velocity)
    if len(ri) and len(ei):
        _, _, vel_f1, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ri, rp_hz, rv, ei, ep_hz, ev, onset_tolerance=ONSET_TOL, offset_ratio=None)
        vel_f1 = float(vel_f1)
        matches = mir_eval.transcription.match_notes(
            ri, rp_hz, ei, ep_hz, onset_tolerance=ONSET_TOL, offset_ratio=None)
    else:
        vel_f1, matches = 0.0, []

    if matches:
        rvm = np.array([rv[i] for i, _ in matches], float)
        evm = np.array([ev[j] for _, j in matches], float)
        vel_mae = float(np.mean(np.abs(rvm - evm)))
        vel_rho = float(spearmanr(rvm, evm).statistic) if len(matches) > 2 else None
        onset_mae_ms = float(np.mean(np.abs(
            np.array([ri[i, 0] for i, _ in matches]) - np.array([ei[j, 0] for _, j in matches])))) * 1000
    else:
        vel_mae = vel_rho = onset_mae_ms = None

    # pedal frame F1
    rg, eg = _pedal_frames(ref_path), _pedal_frames(est_path)
    tp = int(np.sum(rg & eg)); fp = int(np.sum(~rg & eg)); fn = int(np.sum(rg & ~eg))
    pedal_f1 = _f1(tp, fp, fn) if (rg.any() or eg.any()) else None

    return {"onset_f1": onset_f1, "onset_offset_f1": onoff_f1, "velocity_f1": vel_f1,
            "velocity_mae": vel_mae, "velocity_spearman": vel_rho,
            "onset_mae_ms": onset_mae_ms, "pedal_frame_f1": pedal_f1,
            "n_ref_notes": int(len(ri)), "n_est_notes": int(len(ei)), "n_matched": len(matches)}


def _agg(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def main():
    manifest = json.loads((BENCH / "manifest.json").read_text())
    models = {k: d for k, d in MODEL_DIRS.items() if d.exists() and any(d.glob("*.mid"))}
    print(f"models present: {list(models)}   clips: {len(manifest)}\n", flush=True)

    per_model = {}
    for model, mdir in models.items():
        rows = []
        for m in manifest:
            gt = GT / f"{m['seg']}.mid"
            est = mdir / f"{m['seg']}.mid"
            if not gt.exists() or not est.exists():
                raise FileNotFoundError(
                    f"incomplete paired benchmark for {model}/{m['seg']}: "
                    f"gt_exists={gt.exists()} est_exists={est.exists()}"
                )
            try:
                rows.append(evaluate(gt, est))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"evaluation failed for {model}/{m['seg']}") from exc
        keys = ["onset_f1", "onset_offset_f1", "velocity_f1", "velocity_mae",
                "velocity_spearman", "onset_mae_ms", "pedal_frame_f1"]
        per_model[model] = {"n_clips": len(rows), **{k: _agg([r[k] for r in rows]) for k in keys}}

    OUT.write_text(json.dumps({"n_clips": len(manifest), "per_model": per_model}, indent=2, default=float))

    print("=== STAGE 0: EXPRESSIVE TRANSCRIPTION BENCHMARK (MAESTRO test, 30s clips) ===")
    hdr = ["model", "onsetF1", "on+offF1", "velF1", "velMAE↓", "velRho↑", "onsetMS↓", "pedalF1"]
    print("  " + "  ".join(f"{h:>9}" for h in hdr))
    for model, r in per_model.items():
        def f(k, nd=3):
            return "  None  " if r[k] is None else f"{r[k]:.{nd}f}"
        print("  " + "  ".join(f"{x:>9}" for x in [
            model, f("onset_f1"), f("onset_offset_f1"), f("velocity_f1"),
            f("velocity_mae", 1), f("velocity_spearman"), f("onset_mae_ms", 1), f("pedal_frame_f1")]))
    print(f"\n  velMAE/velRho/pedalF1/onsetMS are the EXPRESSIVE metrics (the decision drivers).")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
