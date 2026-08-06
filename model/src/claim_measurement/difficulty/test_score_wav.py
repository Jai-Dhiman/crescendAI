"""Tests for score_wav (#166 / #104 S1) -- the MIREX submission seam.

Everything here is CPU-only and offline: the transcriber and the embedder are
injected, exactly as train_fold.py's loader_factory pattern does, so the fast
suite never needs Transkun, the 1.6GB checkpoint, or the isolated env.

The ONE test that does need all three -- a real WAV scored end to end through
the real backbone with no cache anywhere -- lives in
test_score_wav_integration.py and is opt-in.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np
import pytest

from claim_measurement.difficulty.score_wav import (
    ALPHAS,
    fit_ridge_head,
    read_ridge_head,
    score_wav,
    score_wav_or_fallback,
    write_ridge_head,
)

_NOTES = [
    {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
    {"pitch": 64, "onset": 0.5, "offset": 1.0, "velocity": 90},
    {"pitch": 67, "onset": 1.0, "offset": 1.75, "velocity": 70},
]


def _training_data(n=120, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    y = X @ rng.normal(size=dim) + rng.normal(scale=0.1, size=n)
    return X, y


# --------------------------------------------------------------------------
# The head: the only stateful object the research harness never persisted.
# --------------------------------------------------------------------------


def test_serialized_head_reproduces_the_sklearn_pipeline_it_was_fit_from():
    """The deployed head must BE the measured head. ft_eval.py scores through
    StandardScaler + RidgeCV; the container scores through four numpy arrays.
    If those two ever disagree, every tau-c we measured describes a system that
    is not the one being submitted -- so pin the equivalence rather than
    trusting that `(x - mean) / scale @ coef + intercept` is what sklearn does.

    Plain arrays rather than a pickled pipeline on purpose: a pickle is welded
    to the sklearn version that wrote it, and the container pins its own env.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _training_data()
    reference = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    reference.fit(X, y)

    head = fit_ridge_head(X, y, fallback_score=5.0)

    ours = np.array([head.predict(row) for row in X])
    np.testing.assert_allclose(ours, reference.predict(X), rtol=1e-9, atol=1e-9)


def test_head_round_trips_through_the_npz_it_is_shipped_as(tmp_path):
    X, y = _training_data()
    head = fit_ridge_head(X, y, fallback_score=5.0)
    path = tmp_path / "ridge_head.npz"

    write_ridge_head(path, head)
    reloaded = read_ridge_head(path)

    np.testing.assert_array_equal(reloaded.mean, head.mean)
    np.testing.assert_array_equal(reloaded.scale, head.scale)
    np.testing.assert_array_equal(reloaded.coef, head.coef)
    assert reloaded.intercept == head.intercept
    assert reloaded.fallback_score == head.fallback_score
    assert reloaded.predict(X[0]) == head.predict(X[0])


def test_head_rejects_an_embedding_of_the_wrong_width():
    """A dimension mismatch between backbone and head means the wrong adapter
    or the wrong head was staged into the container. Silently broadcasting it
    would emit a confident number from a mismatched system; inside the
    container this raises and the caller converts it to the loud fallback."""
    X, y = _training_data(dim=8)
    head = fit_ridge_head(X, y, fallback_score=5.0)

    with pytest.raises(ValueError, match="expects 8"):
        head.predict(np.zeros(4))


def test_a_constant_feature_does_not_divide_by_zero():
    """StandardScaler maps zero-variance columns to scale 1.0, not 0.0. A
    mean-pooled embedding dimension that never varies across the corpus is
    entirely plausible, and dividing by its 0.0 std would emit nan."""
    X, y = _training_data()
    X[:, 3] = 7.0

    head = fit_ridge_head(X, y, fallback_score=5.0)

    assert np.all(head.scale > 0)
    assert np.isfinite(head.predict(X[0]))


# --------------------------------------------------------------------------
# The seam: WAV -> transcribe -> MIDI -> embed -> head -> float.
# --------------------------------------------------------------------------


def _fake_transcribe(calls=None):
    def transcribe(wav_path):
        if calls is not None:
            calls.append(wav_path)
        return list(_NOTES), []

    return transcribe


def test_score_wav_returns_one_float_from_a_wav_path(tmp_path):
    """The whole submission contract in one line: a path to a WAV in, a single
    real-valued score out."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"not really a wav; the transcriber is injected")
    X, y = _training_data(dim=4)
    head = fit_ridge_head(X, y, fallback_score=5.0)

    score = score_wav(wav, transcribe=_fake_transcribe(),
                      embed=lambda midi: np.arange(4, dtype=np.float32),
                      head=head)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_the_transcribed_notes_reach_the_embedder_as_a_real_midi_file(tmp_path):
    """MoonBeam's MusicTokenizer consumes a FILE, so the seam has to
    materialise the note list. Assert the encoder sees the transcription
    itself, not an empty or placeholder MIDI -- a silent stand-in here would
    make every piece score the same and still return a plausible float."""
    from claim_measurement.difficulty.psyllabus import notes_from_midi_bytes

    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"stub")
    X, y = _training_data(dim=4)
    seen = []

    def embed(midi_path):
        seen.append(notes_from_midi_bytes(midi_path.read_bytes()))
        return np.arange(4, dtype=np.float32)

    score_wav(wav, transcribe=_fake_transcribe(), embed=embed,
              head=fit_ridge_head(X, y, fallback_score=5.0))

    assert len(seen) == 1
    assert [n["pitch"] for n in seen[0]] == [60, 64, 67]


def test_the_intermediate_midi_is_not_left_behind(tmp_path):
    """The container scores a whole test set in one process; leaking a temp
    MIDI per item would be a slow disk-fill rather than a visible failure."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"stub")
    X, y = _training_data(dim=4)
    captured = []

    def embed(midi_path):
        captured.append(midi_path)
        return np.arange(4, dtype=np.float32)

    score_wav(wav, transcribe=_fake_transcribe(), embed=embed,
              head=fit_ridge_head(X, y, fallback_score=5.0))

    assert captured and not captured[0].exists()


def test_scoring_the_same_wav_twice_gives_a_bit_identical_score(tmp_path):
    """MIREX scores each recording independently and ranks by tau-c; a score
    that moves between runs is a score we cannot reason about. Transkun is
    deterministic and build_fold_embedder calls .eval(), so the only way this
    breaks is a random window or a dropout left live -- both of which have
    already bitten this pipeline once."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"stub")
    X, y = _training_data(dim=4)
    head = fit_ridge_head(X, y, fallback_score=5.0)
    embed = lambda midi: np.arange(4, dtype=np.float32)  # noqa: E731

    first = score_wav(wav, transcribe=_fake_transcribe(), embed=embed, head=head)
    second = score_wav(wav, transcribe=_fake_transcribe(), embed=embed, head=head)

    assert first == second


# --------------------------------------------------------------------------
# The failure-policy INVERSION -- container-only. >5% item failures excludes
# the submission from ranking, so a raise here costs the whole submission.
# --------------------------------------------------------------------------


def test_a_failed_item_emits_the_fallback_and_never_raises(tmp_path, capsys):
    """model/CLAUDE.md mandates loud failure over silent fallback, and inside
    this container that rule inverts: an uncaught exception on >5% of items
    excludes the submission from ranking entirely. Loud AND non-fatal -- the
    log is what keeps this from being the silent fallback the rule forbids."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"stub")
    X, y = _training_data(dim=4)
    head = fit_ridge_head(X, y, fallback_score=5.0)

    def exploding_transcribe(wav_path):
        raise RuntimeError("transkun exited 1")

    score, ok = score_wav_or_fallback(
        wav, transcribe=exploding_transcribe,
        embed=lambda midi: np.arange(4, dtype=np.float32), head=head)

    assert ok is False
    assert score == 5.0
    err = capsys.readouterr().err
    assert "SCORE_FAILURE" in err and "transkun exited 1" in err


def test_a_successful_item_reports_ok_and_logs_nothing(tmp_path, capsys):
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"stub")
    X, y = _training_data(dim=4)
    head = fit_ridge_head(X, y, fallback_score=5.0)

    score, ok = score_wav_or_fallback(
        wav, transcribe=_fake_transcribe(),
        embed=lambda midi: np.arange(4, dtype=np.float32), head=head)

    assert ok is True
    assert score != 5.0
    assert capsys.readouterr().err == ""


def test_a_non_finite_score_counts_as_a_failure(tmp_path, capsys):
    """A nan reaching the output file is worse than an exception: MIREX would
    read it as a real prediction. Treat it as a failure and fall back."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"stub")
    X, y = _training_data(dim=4)
    head = fit_ridge_head(X, y, fallback_score=5.0)

    score, ok = score_wav_or_fallback(
        wav, transcribe=_fake_transcribe(),
        embed=lambda midi: np.full(4, np.nan, dtype=np.float32), head=head)

    assert ok is False
    assert score == 5.0
    assert "SCORE_FAILURE" in capsys.readouterr().err


def test_an_empty_transcription_is_a_failure_not_a_zero_vector(tmp_path, capsys):
    """A WAV that transcribes to no notes at all (silence, non-piano audio) is
    a real item on the pathological tail. MoonBeam would still emit some
    embedding for an empty MIDI, and that embedding means nothing -- so this
    is a failure to fall back from, not a score to report."""
    wav = tmp_path / "piece.wav"
    wav.write_bytes(b"stub")
    X, y = _training_data(dim=4)
    head = fit_ridge_head(X, y, fallback_score=5.0)

    score, ok = score_wav_or_fallback(
        wav, transcribe=lambda p: ([], []),
        embed=lambda midi: np.arange(4, dtype=np.float32), head=head)

    assert ok is False
    assert score == 5.0
    assert "no notes" in capsys.readouterr().err


def test_load_scorer_rejects_an_unknown_failure_mode(tmp_path):
    """A typo'd --on-failure must not silently pick a policy. Which of the two
    behaviours is live is the difference between shipping a bug and being
    excluded from ranking, so it is not a value to guess at."""
    from claim_measurement.difficulty.score_wav import load_scorer

    with pytest.raises(ValueError, match="on_failure"):
        load_scorer(tmp_path, checkpoint=None, repo_root=None,
                    model_config=None, on_failure="fallbcak")


# --------------------------------------------------------------------------
# The isolated env, which is where three of this phase's launches died.
# --------------------------------------------------------------------------


def test_script_header_declares_every_dep_this_module_and_its_imports_need():
    """`uv run --script` builds the isolated env from this file's `# /// script`
    header ALONE, and the container reproduces that env. A module-scope import
    the header does not declare fails at startup; a LAZY one fails mid-run,
    after real work. Same check audio_emb_extract.py carries, same reason.
    """
    import ast
    import re
    import sys
    from importlib import import_module
    from pathlib import Path

    # import_module rather than `from ... import score_wav`: the module and its
    # main function share a name, and this must resolve to the module.
    module = import_module("claim_measurement.difficulty.score_wav")

    path = Path(module.__file__).resolve()
    module_dir = path.parent
    block = re.search(r"# /// script\n(.*?)# ///", path.read_text(), re.DOTALL).group(1)
    declared = {re.split(r"[<>=!\[]", d)[0].strip().lower().replace("-", "_")
                for d in re.findall(r'"([^"]+)"', block)}
    declared |= {"sklearn" if d == "scikit_learn" else d for d in declared}

    chain = ["score_wav.py", "audio_emb_extract.py", "bakeoff_cv.py",
             "bakeoff_npz.py", "train_fold.py", "realaudio_check.py"]
    for filename in chain:
        tree = ast.parse((module_dir / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                # transkun_cli is apps/inference/amt/transkun_cli.py put on
                # sys.path by _import_transcribe_wav, not a distribution --
                # same category as the fork's vendored transformers. Its own
                # module-scope deps (numpy, pretty_midi, soundfile) ARE
                # declared above. It shells out to `uv run --with transkun`,
                # so the CONTAINER needs uv plus a warm uv cache; that is a
                # Dockerfile requirement, not a header one.
                if (root in ("__future__", "claim_measurement", "transformers",
                             "bakeoff_cv", "ranking_loss", "trackio",
                             "transkun_cli")
                        or root in sys.stdlib_module_names):
                    continue
                assert root in declared, (
                    f"{filename} imports {root!r}, but score_wav.py's "
                    f"`# /// script` header does not declare it. The isolated "
                    f"env would die on it. Declared: {sorted(declared)}")


def test_the_alphas_grid_matches_the_one_every_measurement_used():
    """The head deployed in the container must be fit with the same
    hyperparameter search the 0.8395 was measured through. A different grid is
    a different model wearing the same number."""
    from claim_measurement.difficulty import ft_eval

    np.testing.assert_array_equal(ALPHAS, ft_eval.ALPHAS)
