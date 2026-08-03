"""Paired aria-vs-transkun substrate comparison (#101 FRONT 8d) -- pure, no I/O."""
from __future__ import annotations

import pytest

from claim_measurement.dynamics_supply.compare_substrates import (
    _exact_mcnemar_p,
    compare,
    join_records,
    mcnemar,
    polarity_shift,
    transition_matrix,
)


def _rec(seg, pol, verdict, committed, d):
    label = {"+": "loud", "-": "soft", "neutral": "balanced"}[pol]
    return {"segment": seg, "gt_polarity": pol, "gt_label": label,
            "verdict": verdict, "committed": committed, "amt_d": d}


def test_join_raises_when_segment_sets_differ():
    aria = [_rec("a", "+", "SUPPORTED", True, 10.0)]
    tk = [_rec("b", "+", "SUPPORTED", True, 10.0)]
    with pytest.raises(ValueError, match="paired design broken"):
        join_records(aria, tk)


def test_join_carries_gt_polarity_from_aria_only():
    # gt_polarity is GT-derived, so aria's value is authoritative for the pair.
    aria = [_rec("s1", "-", "SUPPORTED", True, -9.0)]
    tk = [_rec("s1", "-", "REFUTED", True, 2.0)]
    (p,) = join_records(aria, tk)
    assert p["gt_polarity"] == "-"
    assert p["aria_verdict"] == "SUPPORTED"
    assert p["transkun_verdict"] == "REFUTED"
    assert p["aria_d"] == -9.0 and p["transkun_d"] == 2.0


def test_transition_matrix_counts_flips_and_abstains():
    aria = [_rec("s1", "+", "SUPPORTED", True, 10.0),
            _rec("s2", "-", "SUPPORTED", True, -9.0),
            _rec("s3", "neutral", "SUPPORTED", False, 1.0)]   # abstain
    tk = [_rec("s1", "+", "SUPPORTED", True, 12.0),
          _rec("s2", "-", "REFUTED", True, 3.0),               # flip SUP->REF
          _rec("s3", "neutral", "SUPPORTED", True, 4.0)]       # abstain->SUP
    pairs = join_records(aria, tk)
    m = transition_matrix(pairs)
    assert m["SUPPORTED->SUPPORTED"] == 1
    assert m["SUPPORTED->REFUTED"] == 1
    assert m["ABSTAIN->SUPPORTED"] == 1


def test_exact_mcnemar_symmetric_and_bounded():
    assert _exact_mcnemar_p(0, 0) == 1.0
    assert _exact_mcnemar_p(3, 3) == pytest.approx(_exact_mcnemar_p(3, 3))
    # all discordance one way -> smallest p; b=5,c=0 -> 2*0.5^5 = 0.0625
    assert _exact_mcnemar_p(5, 0) == pytest.approx(0.0625)
    assert _exact_mcnemar_p(2, 1) <= 1.0


def test_mcnemar_only_uses_committed_in_both():
    pairs = join_records(
        [_rec("s1", "+", "SUPPORTED", True, 10.0),
         _rec("s2", "-", "SUPPORTED", True, -9.0),
         _rec("s3", "+", "SUPPORTED", False, 1.0)],   # aria abstains -> excluded
        [_rec("s1", "+", "REFUTED", True, 3.0),        # discordant b
         _rec("s2", "-", "SUPPORTED", True, -8.0),     # concordant
         _rec("s3", "+", "SUPPORTED", True, 12.0)],
    )
    mc = mcnemar(pairs)
    assert mc["n_committed_both"] == 2
    assert mc["discordant_aria_sup_tk_ref"] == 1
    assert mc["discordant_aria_ref_tk_sup"] == 0


def test_polarity_shift_reports_louder_transkun_on_soft():
    pairs = join_records(
        [_rec("s1", "-", "SUPPORTED", True, -9.0), _rec("s2", "-", "SUPPORTED", True, -7.0)],
        [_rec("s1", "-", "REFUTED", True, 3.0), _rec("s2", "-", "SUPPORTED", True, -1.0)],
    )
    shift = polarity_shift(pairs)
    assert shift["-"]["label"] == "soft"
    assert shift["-"]["n"] == 2
    # tk reads louder (less negative d): (3-(-9) + (-1)-(-7))/2 = (12+6)/2 = 9.0
    assert shift["-"]["mean_shift_tk_minus_aria"] == pytest.approx(9.0)


def test_compare_end_to_end_rate_delta():
    def _result(rate, ncomm, half, recs):
        return {"faithfulness_rate": rate, "n_committed": ncomm,
                "ci95": {"half_width": half}, "gd_pass": True, "per_segment": recs}
    aria = _result(0.90, 2, 0.04,
                   [_rec("s1", "+", "SUPPORTED", True, 10.0), _rec("s2", "-", "REFUTED", True, 2.0)])
    tk = _result(1.00, 2, 0.03,
                 [_rec("s1", "+", "SUPPORTED", True, 12.0), _rec("s2", "-", "SUPPORTED", True, -8.0)])
    out = compare(aria, tk)
    assert out["n_paired_segments"] == 2
    assert out["rate_delta_tk_minus_aria"] == pytest.approx(0.10)
    assert out["mcnemar_committed_both"]["discordant_aria_ref_tk_sup"] == 1
