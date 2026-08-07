# model/tests/follower_eval/test_ood_eval.py
"""Unit tests for the OOD per-factor table (#148).

The claim this module makes is that its metrics come from ``asap_eval``
UNCHANGED, so that #148's numbers are subtractable from Track A's. That claim
is testable two ways and both are exercised here: the symbols really are the
imported ones, and the module contains no metric arithmetic of its own."""

from __future__ import annotations

import inspect

import pytest
from follower_eval import asap_eval, ood_eval


def _arm(take, level, med, w1, n=100):
    return ood_eval.ArmResult(
        take_id=take,
        factor="note_source",
        level=level,
        n_beats_eval=n,
        median_abs_err_beats=med,
        p90_abs_err_beats=None if med is None else med * 2,
        within_1beat_frac=w1,
    )


# --- the "imported unchanged" claim -----------------------------------------


def test_the_metric_core_is_the_imported_symbol_not_a_local_copy():
    """If follow_window were reimplemented here, this identity check fails and
    the subtractability claim in the module docstring is false."""
    assert ood_eval.follow_window is asap_eval.follow_window


def test_ood_eval_defines_no_metric_of_its_own():
    """A guard against drift by accretion: someone adding a 'quick local
    median' here would silently break comparability with Track A."""
    source = inspect.getsource(ood_eval)
    for banned in (
        "def _summarize",
        "def _beat_errors",
        "def _pctl",
        "statistics.median",
    ):
        assert banned not in source, f"ood_eval must not define/compute {banned}"


# --- pairing ----------------------------------------------------------------


def test_table_uses_only_takes_present_in_every_level():
    """An unpaired take lets the levels differ on WHICH performances they cover,
    reintroducing exactly the confound the take abstraction removes."""
    results = [
        _arm("a", "midi", 0.001, 0.99),
        _arm("a", "audio", 0.005, 0.97),
        _arm("b", "midi", 0.500, 0.50),  # no audio arm -> must be dropped
    ]
    table = ood_eval.paired_table(results, baseline_level="midi")
    assert table["n_takes_paired"] == 1
    assert table["n_takes_dropped_unpaired"] == 1
    midi = next(r for r in table["rows"] if r["level"] == "midi")
    # 0.500 would have dragged the midi baseline up if 'b' had been kept
    assert midi["median_abs_err_beats"] == pytest.approx(0.001)


def test_a_take_present_but_null_at_one_level_cannot_unpair_the_means():
    """Presence is not a value. A metric is None whenever a window evaluated
    zero beats, which is a plausible phone-channel outcome -- so a take can be
    recorded on every arm and still have nothing to contribute at one of them.

    Averaging each level over whatever it happened to have non-None compares
    DIFFERENT sets of takes, and can invert the sign: here 'b' follows on clean
    and yields nothing on phone, so its large clean error stays in the baseline
    while its absent phone error leaves the other mean, and the degraded channel
    reads BETTER than the clean one."""
    results = [
        _arm("a", "clean", 0.10, 0.90),
        _arm("b", "clean", 0.90, 0.10),
        _arm("a", "phone", 0.20, 0.80),
        _arm("b", "phone", None, None, n=0),  # present, but no evaluable beat
    ]
    table = ood_eval.paired_table(results, baseline_level="clean")
    phone = next(r for r in table["rows"] if r["level"] == "phone")

    # Both means must rest on take 'a' alone -- the only take with a value at
    # every level -- so the phone channel reads WORSE, which it is.
    assert table["n_takes_dropped_null_median"] == 1
    assert phone["n_takes_median"] == 1
    assert phone["delta_median_beats"] == pytest.approx(0.10)
    assert phone["delta_within_1beat_pp"] == pytest.approx(-10.0)


def test_a_level_with_no_usable_take_reports_none_rather_than_crashing():
    """The report must still print. A formatter that raises tells you less than
    one that shows the gap."""
    results = [
        _arm("a", "clean", 0.10, 0.90),
        _arm("a", "phone", None, None, n=0),
    ]
    table = ood_eval.paired_table(results, baseline_level="clean")
    assert table["n_takes_dropped_null_median"] == 1
    for row in table["rows"]:
        assert row["median_abs_err_beats"] is None
        assert row["delta_median_beats"] is None
    rendered = ood_eval._format({"factor": "channel", "table": table,
                                 "metric_core": "x", "failures": []})
    assert "--" in rendered


def test_deltas_are_measured_against_the_named_baseline():
    results = [
        _arm("a", "midi", 0.001, 0.99),
        _arm("a", "audio", 0.005, 0.97),
    ]
    rows = {r["level"]: r for r in ood_eval.paired_table(results, "midi")["rows"]}
    assert rows["midi"]["delta_median_beats"] == pytest.approx(0.0)
    assert rows["audio"]["delta_median_beats"] == pytest.approx(0.004)
    assert rows["audio"]["delta_within_1beat_pp"] == pytest.approx(-2.0)


def test_table_raises_when_no_take_is_paired():
    results = [_arm("a", "midi", 0.001, 0.99), _arm("b", "audio", 0.005, 0.97)]
    with pytest.raises(ood_eval.OodEvalError, match="unpaired"):
        ood_eval.paired_table(results, baseline_level="midi")


def test_table_raises_on_an_unknown_baseline_level():
    results = [_arm("a", "midi", 0.001, 0.99), _arm("a", "audio", 0.005, 0.97)]
    with pytest.raises(ood_eval.OodEvalError, match="baseline level"):
        ood_eval.paired_table(results, baseline_level="phone_near")


def test_a_third_level_slots_in_without_touching_the_table_code():
    """When phone channels arrive they are just more levels. If this needs a
    code change, the abstraction failed."""
    results = [
        _arm("a", "midi", 0.001, 0.99),
        _arm("a", "audio", 0.005, 0.97),
        _arm("a", "phone_near", 0.030, 0.90),
    ]
    rows = {r["level"]: r for r in ood_eval.paired_table(results, "midi")["rows"]}
    assert set(rows) == {"midi", "audio", "phone_near"}
    assert rows["phone_near"]["delta_within_1beat_pp"] == pytest.approx(-9.0)
