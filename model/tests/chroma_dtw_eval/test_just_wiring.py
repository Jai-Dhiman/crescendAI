import shutil
import subprocess


def test_just_chroma_eval_verify_exits_zero():
    if shutil.which("just") is None:
        import pytest; pytest.skip("just not installed")
    # Use smoke recipe: exercises sampler + pseudo-truth + aggregator without real audio.
    # Full chroma-eval-verify (no --skip-dtw) requires real audio; tested separately.
    result = subprocess.run(
        ["just", "chroma-eval-verify-smoke"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}; stderr={result.stderr}"
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    # Stdout may include 'just' echoing -- find the float line.
    floats = [ln for ln in lines if _is_float(ln)]
    assert floats, f"no float line in stdout: {result.stdout!r}"


def _is_float(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False
