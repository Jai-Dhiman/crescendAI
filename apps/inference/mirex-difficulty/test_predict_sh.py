"""#171: predict.sh's argument handling, tested without a built image.

predict.sh's whole job is translating whatever MIREX hands the container into
one score_wav.py invocation. test_container.py covers the real thing but needs
a ~21GB image, so it cannot be run per-commit -- and the invocation shape is
precisely the part we are guessing at, because the task page specifies the
contract and not the calling convention.

So these tests put a FAKE `uv` earlier on PATH. It records its argv and writes
one '<path>\\t<n>' line per input to the file named by --out, which is exactly
what score_wav.py does. No production seam is added for testability: the shim
substitutes at the process boundary predict.sh already goes through.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PREDICT_SH = Path(__file__).with_name("predict.sh")

FAKE_UV = """#!/usr/bin/env bash
# Stands in for `uv run --script score_wav.py`: read --wav-list, write
# '<path>\\t<index>' lines to --out. The index makes score order observable.
printf '%s\\n' "$@" > "$ARGV_LOG"
out=""; list=""; prev=""
for a in "$@"; do
    case "$prev" in
        --out) out="$a" ;;
        --wav-list) list="$a" ;;
    esac
    prev="$a"
done
echo "MusicTokenizer vocabulary noise that must never reach stdout"
i=0
while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] || continue
    i=$((i + 1))
    printf '%s\\t%s.0\\n' "$line" "$i" >> "$out"
done < "$list"
"""


@pytest.fixture
def run(tmp_path):
    """Run predict.sh with a fake `uv`, returning (CompletedProcess, argv)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "uv"
    fake.write_text(FAKE_UV)
    fake.chmod(0o755)
    argv_log = tmp_path / "argv.log"

    def _run(args, stdin=None):
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "ARGV_LOG": str(argv_log),
            "MIREX_MODEL_DIR": "/app/model",
            "MIREX_ON_FAILURE": "raise",
            "MIREX_DEVICE": "cpu",
            "MIREX_CHECKPOINT": "/app/moonbeam/ckpt.pt",
            "MIREX_REPO_ROOT": "/app/moonbeam/repo",
            "MIREX_MODEL_CONFIG": "/app/moonbeam/repo/model_config.json",
        }
        proc = subprocess.run(
            ["bash", str(PREDICT_SH), *args], capture_output=True, text=True,
            input=stdin, env=env, timeout=60)
        argv = argv_log.read_text().splitlines() if argv_log.exists() else []
        return proc, argv

    return _run


def _wavs(tmp_path, *names):
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"RIFF")
    return [str(tmp_path / name) for name in names]


def test_a_single_wav_path_yields_one_bare_score(run, tmp_path):
    """The contract at its narrowest: one WAV path in, one float out, and
    nothing else on stdout."""
    (wav,) = _wavs(tmp_path, "a.wav")

    proc, _ = run([wav])

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "1.0\n"


def test_several_wav_paths_keep_input_order(run, tmp_path):
    """A judge pairing scores to inputs positionally is the likeliest batch
    convention, so order must survive the normalisation into a list file."""
    wavs = _wavs(tmp_path, "c.wav", "a.wav", "b.wav")

    proc, argv = run(wavs)

    assert proc.stdout == "1.0\n2.0\n3.0\n"
    assert argv[argv.index("--wav-list") + 1].startswith("/")
    listed = Path(argv[argv.index("--wav-list") + 1])
    assert not listed.exists(), "the temp list must be cleaned up on exit"


def test_a_directory_expands_recursively_and_in_sorted_order(run, tmp_path):
    """A test set handed over as a directory must not resolve to zero files
    because of nesting -- an empty run looks like a clean one."""
    _wavs(tmp_path, "b.wav", "nested/a.wav", "notes.txt")
    (tmp_path / "notes.txt").write_text("not audio")

    proc, _ = run([str(tmp_path)])

    assert proc.stdout == "1.0\n2.0\n", "expected exactly the two .wav files"


def test_an_uppercase_wav_suffix_is_still_found(run, tmp_path):
    _wavs(tmp_path, "LOUD.WAV")

    proc, _ = run([str(tmp_path)])

    assert proc.stdout == "1.0\n"


def test_a_wav_list_file_is_read(run, tmp_path):
    wavs = _wavs(tmp_path, "a.wav", "b.wav")
    listfile = tmp_path / "list.txt"
    listfile.write_text("\n".join(wavs) + "\n")

    proc, _ = run(["--wav-list", str(listfile)])

    assert proc.stdout == "1.0\n2.0\n"


def test_a_manifest_without_a_trailing_newline_keeps_its_last_item(run, tmp_path):
    """bash `read` drops an unterminated final line. A judge-written manifest
    is not guaranteed to end in a newline, and silently scoring n-1 items
    would read as a successful run."""
    wavs = _wavs(tmp_path, "a.wav", "b.wav")
    listfile = tmp_path / "list.txt"
    listfile.write_text("\n".join(wavs))

    proc, _ = run(["--wav-list", str(listfile)])

    assert proc.stdout == "1.0\n2.0\n"


def test_a_crlf_manifest_does_not_produce_bogus_paths(run, tmp_path):
    """score_wav.py builds Path(line) without stripping, so a CR would become
    part of the filename and every item would fail."""
    wavs = _wavs(tmp_path, "a.wav", "b.wav")
    listfile = tmp_path / "list.txt"
    listfile.write_bytes(("\r\n".join(wavs) + "\r\n").encode())

    proc, _ = run(["--wav-list", str(listfile)])

    assert proc.stdout == "1.0\n2.0\n"
    assert "\r" not in proc.stdout


def test_a_list_arrives_on_stdin_via_dash_wav_list(run, tmp_path):
    wavs = _wavs(tmp_path, "a.wav", "b.wav")

    proc, _ = run(["--wav-list", "-"], stdin="\n".join(wavs) + "\n")

    assert proc.stdout == "1.0\n2.0\n"


def test_a_bare_dash_also_reads_the_list_from_stdin(run, tmp_path):
    wavs = _wavs(tmp_path, "a.wav", "b.wav")

    proc, _ = run(["-"], stdin="\n".join(wavs) + "\n")

    assert proc.stdout == "1.0\n2.0\n"


def test_csv_format_pairs_each_score_with_its_path(run, tmp_path):
    """Positional pairing is fragile if the judge interleaves or reorders;
    --format csv makes the association explicit without changing the default."""
    wavs = _wavs(tmp_path, "a.wav", "b.wav")

    proc, _ = run(["--format", "csv", *wavs])

    assert proc.stdout == f"{wavs[0]},1.0\n{wavs[1]},2.0\n"


def test_no_arguments_is_a_usage_error(run):
    proc, _ = run([])

    assert proc.returncode == 2
    assert proc.stdout == ""


def test_an_unknown_format_is_rejected_before_the_model_loads(run, tmp_path):
    """Loading the 839M backbone takes minutes; a typo must not cost them."""
    (wav,) = _wavs(tmp_path, "a.wav")

    proc, argv = run(["--format", "tsv", wav])

    assert proc.returncode == 2
    assert argv == [], "the scorer must not have been invoked"


def test_a_directory_holding_no_wavs_fails_loudly(run, tmp_path):
    """The one case where the repo's loud-failure rule still applies inside the
    container: resolving zero inputs is a wiring mistake, not an item failure,
    so it must not exit 0 having scored nothing."""
    empty = tmp_path / "empty"
    empty.mkdir()

    proc, argv = run([str(empty)])

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert argv == []


def test_stdout_carries_only_scores_even_though_the_scorer_is_noisy(run, tmp_path):
    """The MoonBeam fork's MusicTokenizer prints ~96KB of vocabulary on
    construction. The fake scorer reproduces that; none of it may reach
    stdout, or the judge parses garbage."""
    (wav,) = _wavs(tmp_path, "a.wav")

    proc, _ = run([wav])

    assert proc.stdout == "1.0\n"
    assert "vocabulary noise" not in proc.stdout


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="no shellcheck")
def test_predict_sh_passes_shellcheck():
    proc = subprocess.run(["shellcheck", str(PREDICT_SH)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout
