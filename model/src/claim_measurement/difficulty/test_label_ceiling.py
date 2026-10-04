"""Tests for label_ceiling (#168) -- the GRM that separates latent piece
difficulty from grading-board bias.

The load-bearing test is `test_grm_recovers_synthetic_parameters`: an EM that
converges smoothly to the wrong answer looks exactly like one that works, so
the fit is checked against data generated from KNOWN board parameters.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np
import pytest

from claim_measurement.difficulty.label_ceiling import (
    Judgement,
    _category_probs,
    _tau_b,
    _tau_c,
    board_pair_agreement,
    co_grading_graph,
    fit_grm,
    load_judgements,
    posterior_predictive_agreement,
    simulate_ceiling,
)


def _simulate(n_pieces, boards, rng, rate=0.4):
    """Grades drawn from a known GRM. `boards` maps name -> (a, thresholds)."""
    theta = rng.standard_normal(n_pieces)
    judgements = []
    for i in range(n_pieces):
        for name, (a, b) in boards.items():
            if rng.random() > rate:     # sparse, like the real corpus
                continue
            probs = _category_probs(a, np.asarray(b), np.array([theta[i]]))[:, 0]
            judgements.append(Judgement(piece=f"p{i:04d}", board=name,
                                        grade=int(rng.choice(len(probs),
                                                             p=probs / probs.sum()))))
    return theta, judgements


def test_load_judgements_reports_every_loss():
    """The two list-shaped `related_entries` and the 'null' board must be
    counted in the report, never silently dropped."""
    labels = {
        "a": {"related_entries": {"ABRSM": 3, "RCM": 4}},
        "b": {"related_entries": [{"syllabus": "AMEB", "grade": "3"}]},
        "c": {"related_entries": {"null": 2, "Trinity": 5}},
        "d": {"related_entries": {"LCM": "7"}},
    }
    judgements, report = load_judgements(labels)
    assert report["n_judgements"] == 3
    assert report["skipped_bad_shape"] == 1
    assert report["skipped_non_board"] == 1
    assert report["skipped_bad_grade"] == 1
    assert {j.board for j in judgements} == {"ABRSM", "RCM", "Trinity"}


def test_co_grading_graph_finds_disconnected_boards():
    """Two board clusters that never co-grade a piece cannot be put on one
    severity scale; the fit must be refused rather than silently reported."""
    judgements = [
        Judgement("p1", "A", 1), Judgement("p1", "B", 2),
        Judgement("p2", "C", 3), Judgement("p2", "D", 4),
    ]
    counts, components = co_grading_graph(judgements)
    assert counts[("A", "B")] == 1
    assert counts[("C", "D")] == 1
    assert len(components) == 2
    assert {frozenset(c) for c in components} == {frozenset("AB"), frozenset("CD")}


def test_co_grading_graph_connected_when_a_piece_links_clusters():
    judgements = [
        Judgement("p1", "A", 1), Judgement("p1", "B", 2),
        Judgement("p2", "B", 3), Judgement("p2", "C", 4),
    ]
    _, components = co_grading_graph(judgements)
    assert len(components) == 1


def test_category_probs_are_a_distribution():
    nodes = np.linspace(-3, 3, 7)
    probs = _category_probs(1.4, np.array([-1.0, 0.0, 1.5]), nodes)
    assert probs.shape == (4, 7)
    assert np.allclose(probs.sum(axis=0), 1.0)
    # a harder piece is likelier to land in the top category
    assert probs[-1, -1] > probs[-1, 0]


def test_board_pair_agreement_respects_min_n():
    judgements = [Judgement(f"p{i}", "A", i % 5) for i in range(10)]
    judgements += [Judgement(f"p{i}", "B", i % 5) for i in range(10)]
    assert board_pair_agreement(judgements, min_n=50)["pairs"] == []
    out = board_pair_agreement(judgements, min_n=5)
    assert len(out["pairs"]) == 1
    assert out["pairs"][0]["tau_c"] == pytest.approx(1.0, abs=0.05)


def test_grm_rejects_a_single_category_board():
    judgements = [Judgement(f"p{i}", "A", 3) for i in range(10)]
    judgements += [Judgement(f"p{i}", "B", i % 4) for i in range(10)]
    with pytest.raises(ValueError, match="one grade category"):
        fit_grm(judgements)


def test_grm_recovers_synthetic_parameters():
    """Fit data generated from known boards and check the recovered latent
    trait, discriminations, and severity ORDER match the truth.

    The correlation bar is 0.6, not 0.95, because it is information-limited
    rather than fit-limited: at ~1.6 ordinal judgements per piece (the real
    corpus has 1.73) a continuous latent simply cannot be pinned down.
    `test_grm_is_consistent_with_denser_data` is the test that would catch a
    biased estimator.
    """
    rng = np.random.default_rng(2026)
    truth = {
        "sharp": (2.0, [-1.5, -0.5, 0.5, 1.5]),   # high discrimination
        "blunt": (0.7, [-1.0, 0.0, 1.0]),         # low discrimination
        "harsh": (1.3, [0.5, 1.2, 2.0]),          # severe: high thresholds
        "lax": (1.3, [-2.0, -1.2, -0.5]),         # lenient: low thresholds
    }
    theta, judgements = _simulate(1200, truth, rng)
    fit = fit_grm(judgements)

    keep = [i for i in range(len(theta)) if f"p{i:04d}" in fit.pieces]
    est = np.array([fit.theta[fit.pieces.index(f"p{i:04d}")] for i in keep])
    assert np.corrcoef(est, theta[keep])[0, 1] > 0.6

    assert fit.a["sharp"] > fit.a["blunt"]
    assert np.mean(fit.b["harsh"]) > np.mean(fit.b["lax"])
    assert all(np.all(np.diff(fit.b[b]) > 0) for b in fit.boards)


def test_grm_is_consistent_with_denser_data():
    """The guard against a smoothly-converging wrong answer: with plenty of
    judgements per piece the fit must land close to the generating parameters,
    not merely in the right order. An EM stopped on delta-loglik passed the
    ordering checks above while reporting discrimination 0.64 for a board whose
    true value was 2.5 -- this test is what fails in that case.
    """
    rng = np.random.default_rng(4)
    truth = {f"b{k}": (1.6, [-1.4, -0.4, 0.6, 1.6]) for k in range(8)}
    theta, judgements = _simulate(1500, truth, rng, rate=1.0)
    fit = fit_grm(judgements)

    for b in fit.boards:
        assert fit.a[b] == pytest.approx(1.6, rel=0.25)
        assert np.allclose(fit.b[b], truth[b][1], atol=0.25)
    keep = [i for i in range(len(theta)) if f"p{i:04d}" in fit.pieces]
    est = np.array([fit.theta[fit.pieces.index(f"p{i:04d}")] for i in keep])
    assert np.corrcoef(est, theta[keep])[0, 1] > 0.92


def test_grm_severity_is_not_confounded_with_range_restriction():
    """A board that only ever issues middle grades (like ANZCA, 5..8) must come
    out as scale-COMPRESSED, not as a board that disagrees. Its thresholds must
    span a narrower range than a board using the full scale on identical pieces.
    """
    rng = np.random.default_rng(7)
    truth = {"full": (1.5, [-1.5, -0.5, 0.5, 1.5]),
             "narrow": (1.5, [-0.3, 0.0, 0.3])}
    _, judgements = _simulate(1500, truth, rng)
    fit = fit_grm(judgements)
    full_span = fit.b["full"][-1] - fit.b["full"][0]
    narrow_span = fit.b["narrow"][-1] - fit.b["narrow"][0]
    assert narrow_span < full_span


def test_simulate_ceiling_is_below_one_and_falls_with_noisier_boards():
    """A perfect model scored against a one-board ordinal label cannot reach
    1.0, and a blunter board must push the ceiling lower."""
    rng = np.random.default_rng(11)
    truth = {"sharp": (2.5, [-1.5, -0.5, 0.5, 1.5]),
             "blunt": (0.4, [-1.5, -0.5, 0.5, 1.5])}
    _, judgements = _simulate(800, truth, rng)
    fit = fit_grm(judgements)
    pieces = fit.pieces[:300]
    sharp = simulate_ceiling(fit, pieces, {p: ["sharp"] for p in pieces},
                             n_draws=25, seed=3)
    blunt = simulate_ceiling(fit, pieces, {p: ["blunt"] for p in pieces},
                             n_draws=25, seed=3)
    assert 0.0 < sharp["mean"] < 1.0
    assert blunt["mean"] < sharp["mean"]


def test_simulate_ceiling_rises_with_more_boards_per_piece():
    """Averaging several boards cancels rating noise, so a multi-board label is
    a cleaner readout of the latent trait than any single board's grade."""
    rng = np.random.default_rng(13)
    truth = {"a": (1.2, [-1.0, 0.0, 1.0]), "b": (1.2, [-1.0, 0.0, 1.0]),
             "c": (1.2, [-1.0, 0.0, 1.0])}
    _, judgements = _simulate(900, truth, rng)
    fit = fit_grm(judgements)
    pieces = fit.pieces[:300]
    one = simulate_ceiling(fit, pieces, {p: ["a"] for p in pieces},
                           n_draws=25, seed=5)
    three = simulate_ceiling(fit, pieces, {p: ["a", "b", "c"] for p in pieces},
                             n_draws=25, seed=5)
    assert three["mean"] > one["mean"]


def test_tau_c_is_not_comparable_across_target_granularity():
    """Pins the trap that made 'model vs theta' look worse than 'model vs
    label'. tau-c divides by (m-1)/m for m distinct values, so the SAME
    predictions score ~11/10 higher against an 11-grade label than against a
    continuous latent trait. tau-b has no such factor.
    """
    rng = np.random.default_rng(17)
    latent = rng.standard_normal(600)
    pred = latent + 0.6 * rng.standard_normal(600)
    coarse = np.digitize(latent, np.quantile(latent, np.linspace(0, 1, 12)[1:-1]))

    tau_c_gap = _tau_c(pred, coarse) - _tau_c(pred, latent)
    tau_b_gap = _tau_b(pred, coarse) - _tau_b(pred, latent)
    assert tau_c_gap > 0.04          # the same predictions, a fake +0.04
    assert tau_b_gap < tau_c_gap / 2  # tau-b does not carry the (m-1)/m factor


def test_posterior_predictive_reproduces_observed_agreement():
    """The check that licenses the ceiling: a fit whose simulated boards agree
    far MORE than the real ones would imply a cleaner label than exists."""
    rng = np.random.default_rng(23)
    truth = {f"b{k}": (1.5, [-1.2, -0.3, 0.6, 1.5]) for k in range(6)}
    _, judgements = _simulate(1200, truth, rng, rate=0.9)
    fit = fit_grm(judgements)
    observed = board_pair_agreement(judgements)
    simulated = posterior_predictive_agreement(fit)
    assert simulated["n_weighted"] == pytest.approx(observed["n_weighted"],
                                                    abs=0.12)
