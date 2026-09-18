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
import json
import os
import shutil
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



def _declared_fallback() -> float:
    """The fallback the IMAGE declares, read from its own manifest.

    Not a literal. This was hardcoded to 5.0, which was the fallback of the
    single-adapter model directory the test was written against; the
    five-member submission directory declares 6.0, so the constant would have
    failed the first time the real submission image was tested -- or, worse,
    been "fixed" by editing the number rather than by asking which model was in
    the image. Reading the manifest states the actual invariant: a failed item
    emits the fallback this model declares."""
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", IMAGE,
         "/app/model/manifest.json"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    manifest = json.loads(result.stdout)
    members = manifest.get("members")
    if members:
        return float(members[0]["fallback_score"])
    return float(manifest["fallback_score"])


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


def test_the_container_defaults_to_a_loud_but_non_fatal_fallback():
    """The image ships MIREX_ON_FAILURE=fallback (#171). This asserts the
    DEFAULT baked in, because it is the one setting where the repo's
    loud-failure rule inverts: the contract excludes any submission "failing on
    >5% of items", so a single uncaught exception can cost the submission. Loud
    but non-fatal -- a median score on stdout, SCORE_FAILURE on stderr."""
    result = subprocess.run(
        ["docker", "run", "--rm", "-e", "MIREX_DEVICE=cpu",
         "-v", f"{WAV_DIR}:/wav:ro", IMAGE, "/wav/does_not_exist.wav"],
        capture_output=True, text=True, timeout=TIMEOUT_S)

    assert result.returncode == 0, result.stderr[-3000:]
    assert float(result.stdout.strip()) == pytest.approx(_declared_fallback())
    assert "SCORE_FAILURE" in result.stderr, "the fallback must stay LOUD"


def test_a_real_failure_is_still_reachable_for_debugging():
    """The flip to fallback must not make genuine breakage unobservable: with
    `-e MIREX_ON_FAILURE=raise` a bad item still aborts the run. Without this,
    a wiring mistake during the final rebuild would read as a clean run of
    median scores."""
    result = _run(["/wav/does_not_exist.wav"], on_failure="raise")

    assert result.returncode != 0
    assert result.stdout.strip() == "", "a raised failure must print no score"


def test_a_directory_of_wavs_is_accepted_as_one_invocation():
    """#171: the invocation shape is unspecified by the task page and the
    captains have not answered, so the container accepts a directory as well as
    explicit paths. Batching in one invocation also matters for the 24h budget,
    since every process start reloads the 839M backbone."""
    n_wavs = len(list(WAV_DIR.glob("*.wav")))
    if n_wavs < 2:
        pytest.skip("need at least two WAVs to test directory input")

    result = _run(["--format", "csv", "/wav"])

    assert result.returncode == 0, result.stderr[-3000:]
    lines = result.stdout.strip().split("\n")
    assert len(lines) == n_wavs
    for line in lines:
        path, score = line.rsplit(",", 1)
        assert path.startswith("/wav/")
        assert -5.0 < float(score) < 20.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None,
                    reason="needs ffmpeg to build the format fixtures")
def test_the_contracts_input_formats_all_score_the_same_piece_alike(tmp_path):
    """#171: the task page specifies input as "WAV, 44.1 kHz, mono or stereo".

    Our entire corpus is mono 24 kHz, so until this ran the pipeline had never
    been fed EITHER the contract's sample rate or its channel count -- and
    "transkun resamples internally" was an assumption, not a measurement.

    Scoring the same musical content in all three shapes makes the claim
    falsifiable: a channel-mixing or resampling defect would move the score,
    because the transcription feeding the encoder would change.

    Measured spread on 2026-09-18 was 0.0044 on an 11-level grade scale
    (3.4510 / 3.4548 / 3.4554). The 0.05 bound is an order of magnitude above
    that -- loose enough for transcription nondeterminism, tight enough that
    halving the amplitude by averaging channels wrong would fail it."""
    src = WAV_DIR / HELD_OUT
    audio = tmp_path / "audio"
    audio.mkdir()
    shutil.copy2(src, audio / "a_mono_native.wav")
    for name, args in (("b_mono_44k", ["-ar", "44100", "-ac", "1"]),
                       ("c_stereo_44k", ["-ar", "44100", "-ac", "2"])):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
                        *args, str(audio / f"{name}.wav")],
                       check=True, capture_output=True, timeout=300)

    result = _run(["--format", "csv", "/fmt"],
                  mounts=[(str(audio), "/fmt")])

    assert result.returncode == 0, result.stderr[-3000:]
    lines = result.stdout.strip().split("\n")
    assert len(lines) == 3, f"expected 3 scores, got {lines}"
    by_file = dict(line.rsplit(",", 1) for line in lines)
    scores = [float(v) for v in by_file.values()]
    spread = max(scores) - min(scores)
    assert spread < 0.05, (
        f"sample rate / channel count moved the score by {spread:.4f}: "
        f"{by_file}")
