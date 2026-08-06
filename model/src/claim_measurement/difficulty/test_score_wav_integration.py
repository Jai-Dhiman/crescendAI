"""#166 (#104 S1): the submission contract, executed for real.

Everything in `test_score_wav.py` injects the transcriber and the embedder. This
file injects nothing: a real WAV goes in, Transkun runs, the 839M backbone loads
with fold 0's LoRA adapter, and a float comes out. **No cache is touched** --
the cached transcriptions under `results/phase1_lora/audio_midi_cache/` are what
made the research harness fast and are exactly what MIREX will not provide.

It runs `score_wav.py` as a SUBPROCESS under `uv run --no-project --script`,
which is not an implementation detail but the point: the shared `model/.venv`
carries `tokenizers 0.22.1` and the MoonBeam fork's vendored transformers
hard-requires `>=0.19,<0.20`, so an in-process test would die on import. The
subprocess is the invocation the container will make.

Opt-in, because it costs minutes and 1.6GB of checkpoint. From a worktree,
point CRESCENDAI_DATA_ROOT at the PRIMARY checkout -- `model/data/` is
gitignored, so a worktree's copy is empty and these tests would silently skip:

    CRESCENDAI_RUN_WAV_INTEGRATION=1 \\
    CRESCENDAI_DATA_ROOT=/path/to/primary/model/data \\
    uv run python -m pytest \\
        src/claim_measurement/difficulty/test_score_wav_integration.py -q --no-cov -rs

Build the model directory it reads first (see build_model_dir.py's docstring).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

# CWD-independent: these tests may be collected from anywhere, and the runbook
# and the just recipes do not agree on a working directory.
_DIFFICULTY_DIR = Path(__file__).resolve().parent
# CRESCENDAI_DATA_ROOT exists for the worktree case, which is the normal case
# here: `model/data/` is gitignored, so a `.worktrees/issue-NNN-*` checkout has
# an empty one and every artifact below lives only in the primary checkout.
# Without the override these tests silently SKIP in exactly the tree the work
# is done in.
_DATA_ROOT = Path(os.environ.get(
    "CRESCENDAI_DATA_ROOT", _DIFFICULTY_DIR.parents[2] / "data"))
_MOONBEAM = _DATA_ROOT / "weights" / "moonbeam"
_MODEL_DIR = _DATA_ROOT / "results" / "phase1_lora" / "model_fold0"
_WAV_DIR = _DATA_ROOT / "results" / "amt_gap_curve" / "wav"
_CHECKPOINT = _MOONBEAM / "moonbeam_839M.pt"
_REPO_ROOT = _MOONBEAM / "repo"
# The 839M config. model_config_small.json is the 309M model and would load
# with mismatched keys, which _real_loader refuses rather than measuring.
_MODEL_CONFIG = _REPO_ROOT / "src" / "llama_recipes" / "configs" / "model_config.json"

# A fold-0 HELD-OUT piece: fold 0's adapter never trained on it, so this is not
# a train-on-test smoke -- the contamination #135's 0.824 anchor died of.
_HELD_OUT_WAV = _WAV_DIR / "Bachinskaya_N_The_Old_Cuckoo_Clock.wav"

_TIMEOUT_S = 1800

_REQUIRED = {
    "model dir": _MODEL_DIR / "ridge_head.npz",
    "adapter": _MODEL_DIR / "adapter" / "adapter_config.json",
    "checkpoint": _CHECKPOINT,
    "fork": _MODEL_CONFIG,
    "wav": _HELD_OUT_WAV,
}
_MISSING = [f"{k} ({v})" for k, v in _REQUIRED.items() if not v.exists()]

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CRESCENDAI_RUN_WAV_INTEGRATION") != "1",
        reason="opt-in: costs minutes of CPU and needs the 1.6GB checkpoint; "
               "set CRESCENDAI_RUN_WAV_INTEGRATION=1"),
    pytest.mark.skipif(
        bool(_MISSING),
        # Naming every missing path, not just the first: the worktree case
        # misses all five at once, and one path at a time would read as five
        # separate problems.
        reason="missing artifacts: " + ", ".join(_MISSING)),
]


def _run(wav: Path, out: Path) -> subprocess.CompletedProcess:
    """Invoke score_wav.py exactly as the container will."""
    return subprocess.run(
        ["uv", "run", "--no-project", "--script", "score_wav.py",
         "--model-dir", str(_MODEL_DIR), "--wav", str(wav), "--out", str(out),
         "--checkpoint", str(_CHECKPOINT), "--repo-root", str(_REPO_ROOT),
         "--model-config", str(_MODEL_CONFIG)],
        cwd=_DIFFICULTY_DIR, capture_output=True, text=True, timeout=_TIMEOUT_S)


def _score(wav: Path, out: Path) -> float:
    """Score one WAV and read the result from --out rather than stdout. The
    fork's MusicTokenizer prints its entire vocabulary to stdout on
    construction (~96KB), so stdout is not a usable interface -- a lesson worth
    pinned in a test rather than rediscovered inside a Docker build."""
    result = _run(wav, out)
    assert result.returncode == 0, (
        f"score_wav.py exited {result.returncode}\n"
        f"stderr tail:\n{result.stderr[-4000:]}")
    # The failure-rate line is printed on every run because it is the number
    # that decides whether the submission is ranked at all.
    assert "failures=0" in result.stderr, result.stderr[-2000:]

    line = out.read_text().strip()
    path_part, score_part = line.rsplit("\t", 1)
    assert path_part == str(wav)
    return float(score_part)


def test_a_real_wav_scores_end_to_end_with_no_cache(tmp_path):
    """The submission contract in one assertion: a path to a WAV in, one
    real-valued difficulty score out. Until #166 this had never been executed
    -- the repo held five per-fold adapters and five per-fold ridge heads, a
    measurement apparatus with no path from audio to a number.

    Deliberately asserts only the CONTRACT, not the value. A single piece
    cannot validate a ranking metric, and per #104 no tau-c may be reported
    from a deployed model directory at all.
    """
    score = _score(_HELD_OUT_WAV, tmp_path / "scores.tsv")

    assert isinstance(score, float)
    # The 11-level PSyllabus scale, generously bracketed: a ridge head is not
    # clipped to the scale, so this catches a broken forward pass or a
    # mismatched head, not a mediocre prediction.
    assert -5.0 < score < 20.0, f"score {score} is off any plausible scale"


def test_the_same_wav_scores_identically_in_a_fresh_process(tmp_path):
    """MIREX scores each recording independently and ranks by tau-c; a score
    that moves between runs is a score we cannot reason about. Two SEPARATE
    processes, because the in-process determinism test cannot see a
    seed-dependent load path -- and `lora_dropout=0.05` with a missing
    `.eval()` has already been a P0 on this pipeline once.
    """
    first = _score(_HELD_OUT_WAV, tmp_path / "a.tsv")
    second = _score(_HELD_OUT_WAV, tmp_path / "b.tsv")

    assert first == second, (
        f"same WAV scored {first} then {second} -- something on the path is "
        f"stochastic (dropout left live, or a random window)")


def test_a_wav_that_cannot_be_transcribed_falls_back_instead_of_crashing(tmp_path):
    """The contract excludes any submission failing on >5% of items, so the
    container must survive a corrupt or non-audio file. This is the inversion
    of model/CLAUDE.md's loud-failure rule, and it is confined to the
    container: the fallback is emitted AND logged."""
    broken = tmp_path / "not_audio.wav"
    broken.write_bytes(b"RIFF\x00\x00\x00\x00WAVEthis is not audio")
    out = tmp_path / "scores.tsv"

    result = _run(broken, out)

    assert result.returncode == 0, "a bad item must not fail the whole run"
    assert "SCORE_FAILURE" in result.stderr, "the fallback must be LOUD"
    assert "failures=1" in result.stderr
    score = float(out.read_text().strip().rsplit("\t", 1)[1])
    # Read the expected value out of the head rather than hardcoding it: the
    # fallback is a property of the corpus the head was fit on, so a hardcoded
    # number would silently become wrong the moment the all-data model lands.
    from claim_measurement.difficulty.score_wav import read_ridge_head

    expected = read_ridge_head(_MODEL_DIR / "ridge_head.npz").fallback_score
    assert score == expected, "the fallback is the training corpus's median grade"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--no-cov"]))
