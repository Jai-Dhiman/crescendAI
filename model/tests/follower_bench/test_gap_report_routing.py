"""The gap_report follower-routing seam (#119): additive by default, HMM under --hmm."""
from __future__ import annotations

import argparse
import types

from follower_bench import gap_report
from follower_bench.clip_generator import SynthClip
from follower_bench.follower import ContinuityPrior, DEFAULT_SKIP_PENALTY
from follower_bench.gap_report import _add_cli_args, _follow_for_cell, run_gap_report
from follower_bench.hmm import HmmParams
from follower_bench.score_notes import ScoreNote
from follower_bench.segments import PerfNote
from follower_bench.trajectory import TrueTrajectory


def _tiny():
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64])]
    perf = [PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
            for k, p in enumerate([60, 62, 64])]
    return perf, score


def test_follow_for_cell_routes_additive_by_default_and_hmm_when_requested() -> None:
    perf, score = _tiny()
    prior = ContinuityPrior(skip_penalty=DEFAULT_SKIP_PENALTY)
    add = _follow_for_cell(False, perf, score, prior, HmmParams(), None, (0,))
    assert len(add.matches) == 3
    # additive path leaves confidence None
    assert all(m.confidence is None for m in add.matches)

    hmm = _follow_for_cell(True, perf, score, prior, HmmParams(), None, (0,))
    assert len(hmm.matches) == 3
    # HMM path attaches confidence
    assert all(m.confidence is not None for m in hmm.matches)


def test_gap_report_has_hmm_flag() -> None:
    # The CLI parser must expose --hmm (introspect via a fresh parser build).
    ap = argparse.ArgumentParser()
    _add_cli_args(ap)
    ns = ap.parse_args(["--hmm"])
    assert ns.hmm is True
    # default (no flag) is additive
    assert _add_cli_args(argparse.ArgumentParser()).parse_args([]).hmm is False


# ---- end-to-end use_hmm propagation (challenge caution 4) ----
# run_gap_report -> task tuple -> _run_performance -> _run_cell must carry use_hmm.
# We stub the ASAP data access so this runs in-process with no ASAP dataset, and
# assert the OK outcomes carry a CalibrationStats iff use_hmm=True. A mis-threaded
# tuple would silently run the additive path (calibration None) and fail this.

def _install_synthetic_data(monkeypatch):
    score = [ScoreNote(pitch=p, position=float(i)) for i, p in enumerate([60, 62, 64, 65, 67])]
    perf = tuple(PerfNote(onset=float(k), offset=float(k) + 0.5, pitch=p, velocity=80)
                 for k, p in enumerate([60, 62, 64, 65, 67]))
    clip = SynthClip(asap_piece="synthetic", pathology_type="clean", seed=0, notes=perf,
                     true_trajectory=TrueTrajectory(anchors=((0.0, 0.0), (4.0, 4.0))),
                     event_labels=())
    fake_alignment = types.SimpleNamespace(score_midi_path="unused", midi_score_downbeats=(0.0,))
    monkeypatch.setattr(gap_report, "load_alignment", lambda perf: fake_alignment)
    monkeypatch.setattr(gap_report, "load_score_notes_from_midi", lambda path: list(score))
    monkeypatch.setattr(gap_report, "generate", lambda performance, pathology, seed: clip)
    monkeypatch.setattr(gap_report, "PATHOLOGY_TYPES", ("clean",))


def test_use_hmm_propagates_through_run_gap_report(monkeypatch) -> None:
    _install_synthetic_data(monkeypatch)
    hmm_res = run_gap_report(["synthetic"], [0], workers=1, use_hmm=True)
    assert hmm_res["ok"], "expected at least one scored cell"
    assert all(o.calibration is not None for o in hmm_res["ok"])

    add_res = run_gap_report(["synthetic"], [0], workers=1, use_hmm=False)
    assert add_res["ok"]
    assert all(o.calibration is None for o in add_res["ok"])


# ---- calibration line actually renders (challenge caution 5) ----

def test_format_report_emits_calibration_line_under_hmm(monkeypatch) -> None:
    _install_synthetic_data(monkeypatch)
    result = run_gap_report(["synthetic"], [0], workers=1, use_hmm=True)
    evaluation = gap_report.evaluate_bar(result["aggregates"])
    report = gap_report._format_report(result, evaluation, wall_s=0.1)
    assert "HMM CALIBRATION" in report
    assert "median spearman_rho=" in report

    # Additive path: report is calibration-line-free (byte-neutral to #117).
    add_result = run_gap_report(["synthetic"], [0], workers=1, use_hmm=False)
    add_eval = gap_report.evaluate_bar(add_result["aggregates"])
    add_report = gap_report._format_report(add_result, add_eval, wall_s=0.1)
    assert "HMM CALIBRATION" not in add_report
