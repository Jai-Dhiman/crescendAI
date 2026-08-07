"""Tests for the #166 GPU runtime probe: what gets staged, and what the
measurement means once it comes back.

Everything here is offline and CPU-only. The job script's own `main` is not
exercised -- it needs a GPU, four Hub repos and the 1.6GB checkpoint -- so what
is pinned is the part that would silently produce a wrong ANSWER rather than a
loud failure: the probe's duration spread and the fit read off it.
"""
import json
from pathlib import Path

import pytest

from claim_measurement.difficulty.push_runtime_probe import (
    select_probe_set,
    stage_probe_bundle,
    wav_durations,
)
from claim_measurement.difficulty.runtime_probe import (
    assemble_model_dir,
    build_transcriber,
    fit_runtime,
    project_budget,
)


class _Info:
    def __init__(self, seconds, samplerate=24000):
        self.samplerate = samplerate
        self.frames = int(seconds * samplerate)


def _durations(pairs):
    return [(f"{name}.wav", seconds) for name, seconds in pairs]


# --------------------------------------------------------------------------
# Selection: the spread IS the measurement
# --------------------------------------------------------------------------


def test_selection_spans_the_targets_rather_than_clustering():
    """A probe set bunched near the median fits the fixed term well and the
    slope badly, and the slope is what decides the 24h clause."""
    pool = _durations([(f"p{i}", float(d))
                       for i, d in enumerate([14, 33, 58, 99, 205, 390, 700, 1750,
                                              101, 102, 104])])

    chosen = select_probe_set(pool, targets=(15.0, 100.0, 750.0))

    assert [c["seconds"] for c in chosen] == [14.0, 99.0, 700.0]


def test_selection_never_repeats_a_recording():
    """Two targets that share a nearest neighbour must not both take it --
    that would report more duration spread than the fit actually has."""
    pool = _durations([("a", 100.0), ("b", 900.0)])

    chosen = select_probe_set(pool, targets=(100.0, 101.0, 102.0))

    assert [c["wav"] for c in chosen] == ["a.wav", "b.wav"]


def test_selection_is_ascending_so_an_oom_costs_only_the_longest_item():
    pool = _durations([("long", 1700.0), ("short", 20.0), ("mid", 200.0)])

    chosen = select_probe_set(pool, targets=(1800.0, 15.0, 200.0))

    assert [c["seconds"] for c in chosen] == [20.0, 200.0, 1700.0]


def test_selection_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="no readable WAVs"):
        select_probe_set([])


def test_unreadable_wavs_are_dropped_and_named(tmp_path, capsys):
    """An unopenable file in a bundle whose entire purpose is measurement is
    worse than one fewer probe item."""
    for name in ("good.wav", "bad.wav"):
        (tmp_path / name).write_bytes(b"")

    def _probe(path):
        if path.endswith("bad.wav"):
            raise RuntimeError("not a WAV")
        return _Info(42.0)

    out = wav_durations(tmp_path, _probe)

    assert out == [("good.wav", 42.0)]
    assert "bad.wav" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Staging: the code/ tree has to be importable, not just present
# --------------------------------------------------------------------------


def _module_dir():
    """The real difficulty/ package dir -- staging copies from it, so these
    tests stage the ACTUAL closure rather than a fixture that cannot go
    stale."""
    from claim_measurement.difficulty import push_runtime_probe

    return Path(push_runtime_probe.__file__).resolve().parent


def _wav_dir(tmp_path):
    d = tmp_path / "wav"
    d.mkdir(exist_ok=True)
    (d / "a.wav").write_bytes(b"RIFF")
    return d


def _head_dir(tmp_path):
    d = tmp_path / "head"
    d.mkdir(exist_ok=True)
    (d / "ridge_head.npz").write_bytes(b"")
    (d / "manifest.json").write_text("{}")
    return d


def test_staged_code_is_an_importable_package_tree(tmp_path):
    """score_wav imports `claim_measurement.difficulty.audio_emb_extract`, and
    realaudio_check finds transkun_cli by walking parents for
    apps/inference/amt. A flat pile of .py files satisfies neither."""
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    (wav_dir / "a.wav").write_bytes(b"RIFF")
    head_dir = tmp_path / "head"
    head_dir.mkdir()
    (head_dir / "ridge_head.npz").write_bytes(b"")
    (head_dir / "manifest.json").write_text("{}")
    selection = [{"wav": "a.wav", "seconds": 42.0, "target_s": 30.0}]

    staged = tmp_path / "staged"
    manifest = stage_probe_bundle(wav_dir, head_dir, _module_dir(), staged,
                                  selection)

    assert (staged / "code" / "claim_measurement" / "difficulty"
            / "score_wav.py").exists()
    assert (staged / "code" / "claim_measurement" / "difficulty"
            / "audio_emb_extract.py").exists()
    assert (staged / "code" / "apps" / "inference" / "amt"
            / "transkun_cli.py").exists()
    assert (staged / "wav" / "a.wav").exists()
    assert (staged / "head" / "ridge_head.npz").exists()
    assert manifest["total_audio_s"] == 42.0
    assert json.loads((staged / "manifest.json").read_text())["items"] == selection


def test_staging_refuses_a_head_dir_without_a_ridge_head(tmp_path):
    """An adapter with no head is not a model. Staging one would spend GPU
    time to find that out."""
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    (wav_dir / "a.wav").write_bytes(b"RIFF")
    head_dir = tmp_path / "head"
    head_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="ridge_head.npz"):
        stage_probe_bundle(wav_dir, head_dir, _module_dir(), tmp_path / "s",
                           [{"wav": "a.wav", "seconds": 1.0, "target_s": 1.0}])


# --------------------------------------------------------------------------
# The fit, which is the actual deliverable
# --------------------------------------------------------------------------


def test_fit_recovers_a_known_fixed_term_and_slope():
    rows = [{"audio_s": d, "score_s": 12.0 + 0.25 * d, "ok": True}
            for d in (20.0, 100.0, 400.0, 1600.0)]

    fit = fit_runtime(rows)

    assert fit["fixed_s"] == pytest.approx(12.0, abs=1e-6)
    assert fit["slope_x_realtime"] == pytest.approx(0.25, abs=1e-9)
    assert fit["max_abs_residual_s"] == pytest.approx(0.0, abs=1e-6)


def test_failed_items_are_excluded_from_the_fit():
    """A fallback score returns fast because it skipped the work. Including it
    would report a runtime for a pipeline that did not run."""
    rows = [{"audio_s": 20.0, "score_s": 17.0, "ok": True},
            {"audio_s": 100.0, "score_s": 37.0, "ok": True},
            {"audio_s": 1600.0, "score_s": 0.4, "ok": False}]

    fit = fit_runtime(rows)

    assert fit["n"] == 2
    assert fit["slope_x_realtime"] == pytest.approx(0.25, abs=1e-9)


def test_fit_refuses_to_draw_a_line_through_one_point():
    rows = [{"audio_s": 20.0, "score_s": 17.0, "ok": True},
            {"audio_s": 100.0, "score_s": 37.0, "ok": False}]

    with pytest.raises(ValueError, match="at least 2"):
        fit_runtime(rows)


def test_projection_answers_the_clause_the_contract_actually_states():
    """>5% failures excludes us, and so does missing 24h. The projection is in
    items-per-24h because that is the number the clause is about."""
    fit = {"fixed_s": 10.0, "slope_x_realtime": 0.5}

    p = project_budget(fit, mean_piece_s=100.0)

    assert p["per_item_s"] == pytest.approx(60.0)
    assert p["items_in_budget"] == 1440


def test_probe_binds_the_device_onto_the_transcriber_it_supplies(tmp_path):
    """load_scorer binds --device only onto a transcriber it built itself, so
    a caller that supplies one owns that. Unbound, Transkun stays on CPU and
    the probe measures the exact thing it exists to rule out."""
    staged = tmp_path / "staged"
    stage_probe_bundle(_wav_dir(tmp_path), _head_dir(tmp_path), _module_dir(),
                       staged, [{"wav": "a.wav", "seconds": 1.0, "target_s": 1.0}])

    transcribe = build_transcriber(staged, "cuda")

    assert transcribe.keywords["device"] == "cuda"
    assert transcribe.func.__name__ == "transcribe_wav"


def test_probe_refuses_a_bundle_with_no_transcriber(tmp_path):
    """A GPU job that discovers this at load time has already paid for the
    container. The failure belongs before the spend, and it is loud."""
    empty = tmp_path / "no_code"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="no transcriber"):
        build_transcriber(empty, "cuda")


def test_assemble_model_dir_joins_the_two_hub_halves(tmp_path):
    adapter_src = tmp_path / "hub_adapter"
    (adapter_src / "adapter").mkdir(parents=True)
    (adapter_src / "adapter" / "adapter_config.json").write_text("{}")
    head_src = tmp_path / "hub_head"
    head_src.mkdir()
    (head_src / "ridge_head.npz").write_bytes(b"npz")
    (head_src / "manifest.json").write_text("{}")

    dest = assemble_model_dir(adapter_src, head_src, tmp_path / "model")

    assert (dest / "adapter" / "adapter_config.json").exists()
    assert (dest / "ridge_head.npz").read_bytes() == b"npz"
