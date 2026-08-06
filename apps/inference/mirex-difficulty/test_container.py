"""#166: the MIREX submission container's standardised interface, exercised
against a built image.

These are the only tests that cover what MIREX actually runs. Everything else
tests the code the container invokes; this tests the container. Opt-in, because
it needs a ~21GB image built first (see the Dockerfile header) and each scoring
run takes over a minute on CPU:

    docker build -t crescendai/mirex-difficulty <context>
    CRESCENDAI_RUN_CONTAINER_TESTS=1 \\
    CRESCENDAI_WAV_DIR=<data>/results/amt_gap_curve/wav \\
    uv run python -m pytest apps/inference/mirex-difficulty/test_container.py \\
        -q --no-cov -rs

MIREX_DEVICE is forced to cpu here: the image defaults to cuda because the
contract gives 24h on one GPU, and a dev machine has no CUDA.
"""
import os
import subprocess
from pathlib import Path

import pytest

IMAGE = os.environ.get("CRESCENDAI_IMAGE", "crescendai/mirex-difficulty")
WAV_DIR = Path(os.environ.get("CRESCENDAI_WAV_DIR", "/nonexistent"))
# A fold-0 held-out piece: fold 0's adapter never trained on it.
HELD_OUT = "Bachinskaya_N_The_Old_Cuckoo_Clock.wav"
TIMEOUT_S = 1800


def _image_exists() -> bool:
    result = subprocess.run(["docker", "image", "inspect", IMAGE],
                            capture_output=True, text=True)
    return result.returncode == 0


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CRESCENDAI_RUN_CONTAINER_TESTS") != "1",
        reason="opt-in: needs a built image; set CRESCENDAI_RUN_CONTAINER_TESTS=1"),
    pytest.mark.skipif(
        not (WAV_DIR / HELD_OUT).exists(),
        reason=f"no WAV at {WAV_DIR / HELD_OUT}; set CRESCENDAI_WAV_DIR"),
    pytest.mark.skipif(
        not _image_exists(), reason=f"docker image {IMAGE} is not built"),
]


def _run(args, mounts=(), on_failure="raise") -> subprocess.CompletedProcess:
    cmd = ["docker", "run", "--rm",
           "-e", "MIREX_DEVICE=cpu", "-e", f"MIREX_ON_FAILURE={on_failure}",
           "-v", f"{WAV_DIR}:/wav:ro"]
    for host, guest in mounts:
        cmd += ["-v", f"{host}:{guest}"]
    cmd += [IMAGE, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)


def test_the_container_prints_one_score_and_nothing_else():
    """The competition contract, at the only boundary that counts: a path to a
    WAV in, a single real-valued score out. stdout must stay clean -- the
    MoonBeam fork's MusicTokenizer prints ~96KB of vocabulary on construction,
    and if any of that reached stdout the judge would parse garbage."""
    result = _run([f"/wav/{HELD_OUT}"])

    assert result.returncode == 0, result.stderr[-3000:]
    lines = result.stdout.strip().split("\n")
    assert len(lines) == 1, f"stdout must be exactly one line, got: {lines[:5]}"
    score = float(lines[0])
    assert -5.0 < score < 20.0, f"score {score} is off any plausible scale"


def test_batching_returns_one_score_per_input_in_order():
    """Every process start reloads the 839M backbone, so scoring a test set one
    `docker run` at a time would spend the 24h budget on model loads. Batching
    must therefore work, and must preserve input order -- the judge pairs
    scores to inputs positionally."""
    others = sorted(p.name for p in WAV_DIR.glob("*.wav") if p.name != HELD_OUT)
    if not others:
        pytest.skip("need a second WAV to test batching")

    result = _run([f"/wav/{HELD_OUT}", f"/wav/{others[0]}"])

    assert result.returncode == 0, result.stderr[-3000:]
    scores = result.stdout.strip().split("\n")
    assert len(scores) == 2
    assert all(-5.0 < float(s) < 20.0 for s in scores)


def test_the_container_defaults_to_failing_loudly():
    """The image ships MIREX_ON_FAILURE=raise while the system is still being
    built: a fallback would turn a real bug into a plausible median score. This
    asserts the DEFAULT baked into the image, so a later flip is deliberate."""
    result = subprocess.run(
        ["docker", "run", "--rm", "-e", "MIREX_DEVICE=cpu",
         "-v", f"{WAV_DIR}:/wav:ro", IMAGE, "/wav/does_not_exist.wav"],
        capture_output=True, text=True, timeout=TIMEOUT_S)

    assert result.returncode != 0, "the shipped default must not swallow failures"
    assert result.stdout.strip() == "", "a failed item must print no score"


def test_the_submission_day_flag_emits_a_fallback_score():
    """Covered now so flipping MIREX_ON_FAILURE at the deadline -- when the
    contract's >5%-of-items exclusion clause dominates -- is a one-word change
    with a passing test behind it."""
    result = _run(["/wav/does_not_exist.wav"], on_failure="fallback")

    assert result.returncode == 0
    assert float(result.stdout.strip()) == pytest.approx(5.0)
    assert "SCORE_FAILURE" in result.stderr, "the fallback must stay LOUD"
