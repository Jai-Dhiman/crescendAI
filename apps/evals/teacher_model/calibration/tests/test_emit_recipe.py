# apps/evals/teacher_model/calibration/tests/test_emit_recipe.py
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from teacher_model.calibration.emit_recipe import emit


def _import_recipe(path: Path):
    spec = importlib.util.spec_from_file_location("filter_recipe_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["filter_recipe_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_emit_writes_importable_recipe_with_trusted_subscores(tmp_path: Path):
    calibration_report = {
        "buckets": {
            "ascf_process": "TRUSTED",
            "concrete_artifact_process": "TRUSTED",
            "praise_process": "TRUSTED",
            "autonomy_process": "TRUSTED",
            "scaffolded_process": "TRUSTED",
            "style_process": "TRUSTED",
            "tone_process": "TRUSTED",
            "autonomy_outcome": "TRUSTED",
            "tone_outcome": "CEILING_ARTIFACT",
            "concrete_artifact_outcome": "UNTRUSTED",
            "praise_outcome": "TRUSTED",
        },
        "mean_offset": {sub: 0.0 for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "aggregate_gate_pass": True,
    }
    drift_report = {"intra_rater_kappa": {}, "judge_drift_kappa": {}}

    recipe_path = tmp_path / "filter_recipe.py"
    emit(calibration_report=calibration_report, drift_report=drift_report,
         output_path=recipe_path)

    mod = _import_recipe(recipe_path)
    assert mod.COMPOSITE_PASS_THRESHOLD == 2.5
    # 9 TRUSTED sub-scores get full weight 1.0
    assert set(mod.WEIGHTED_SUB_SCORES.keys()) == {
        "ascf_process", "concrete_artifact_process", "praise_process",
        "autonomy_process", "scaffolded_process", "style_process",
        "tone_process", "autonomy_outcome", "praise_outcome",
    }
    assert all(w == 1.0 for w in mod.WEIGHTED_SUB_SCORES.values())
    # CEILING_ARTIFACT goes to SANITY_FILTERS, not WEIGHTED
    assert "tone_outcome" in mod.SANITY_FILTERS
    # UNTRUSTED is excluded entirely
    assert "concrete_artifact_outcome" not in mod.WEIGHTED_SUB_SCORES
    assert "concrete_artifact_outcome" not in mod.SANITY_FILTERS
    # No bias corrections (no TRUSTED_WITH_OFFSET in this fixture)
    assert mod.BIAS_CORRECTIONS == {}


def test_recipe_filters_synthesis_against_threshold(tmp_path: Path):
    """Round-trip: emit recipe, then a Stage-2-like consumer applies it."""
    calibration_report = {
        "buckets": {sub: "TRUSTED" for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "mean_offset": {sub: 0.0 for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "aggregate_gate_pass": True,
    }
    drift_report = {"intra_rater_kappa": {}, "judge_drift_kappa": {}}
    recipe_path = tmp_path / "filter_recipe_b.py"
    emit(calibration_report=calibration_report, drift_report=drift_report,
         output_path=recipe_path)
    mod = _import_recipe(recipe_path)

    def stage_2_filter(judge_scores_per_sub: dict[str, float]) -> bool:
        weighted = {
            s: judge_scores_per_sub[s] + mod.BIAS_CORRECTIONS.get(s, 0.0)
            for s in mod.WEIGHTED_SUB_SCORES
            if s in judge_scores_per_sub
        }
        if not weighted:
            return False
        composite = sum(weighted.values()) / len(weighted)
        return composite >= mod.COMPOSITE_PASS_THRESHOLD

    high = {s: 3 for s in mod.WEIGHTED_SUB_SCORES}
    low = {s: 1 for s in mod.WEIGHTED_SUB_SCORES}
    assert stage_2_filter(high) is True
    assert stage_2_filter(low) is False


def test_emit_records_bias_correction_for_trusted_with_offset(tmp_path: Path):
    calibration_report = {
        "buckets": {sub: "TRUSTED" for sub in [
            "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "mean_offset": {sub: 0.0 for sub in [
            "ascf_process", "concrete_artifact_process", "praise_process",
            "autonomy_process", "scaffolded_process", "style_process",
            "tone_process", "autonomy_outcome", "tone_outcome",
            "concrete_artifact_outcome", "praise_outcome",
        ]},
        "aggregate_gate_pass": True,
    }
    calibration_report["buckets"]["ascf_process"] = "TRUSTED_WITH_OFFSET"
    calibration_report["mean_offset"]["ascf_process"] = -0.4
    drift_report = {"intra_rater_kappa": {}, "judge_drift_kappa": {}}

    recipe_path = tmp_path / "filter_recipe_offset.py"
    emit(calibration_report=calibration_report, drift_report=drift_report,
         output_path=recipe_path)

    mod = _import_recipe(recipe_path)
    assert "ascf_process" in mod.BIAS_CORRECTIONS
    assert abs(mod.BIAS_CORRECTIONS["ascf_process"] - (-0.4)) < 1e-6
    assert "ascf_process" in mod.WEIGHTED_SUB_SCORES  # still weighted

    # Round-trip: a borderline judge score 2.5 minus 0.4 correction = 2.1 fails;
    # without correction it would pass.
    judge_scores = {s: 2.5 for s in mod.WEIGHTED_SUB_SCORES}
    corrected = {
        s: judge_scores[s] + mod.BIAS_CORRECTIONS.get(s, 0.0)
        for s in mod.WEIGHTED_SUB_SCORES
    }
    composite = sum(corrected.values()) / len(corrected)
    # 10 sub-scores at 2.5 + 1 sub-score at 2.1 = (10*2.5 + 2.1)/11 = 27.1/11 ≈ 2.4636
    assert composite < mod.COMPOSITE_PASS_THRESHOLD
