"""Hard spend-cap behavior: the guard refuses BEFORE the overshooting call."""
from __future__ import annotations

import pytest

from audio_teacher.budget import BudgetExceededError, BudgetGuard


def test_precheck_raises_before_the_overshooting_call():
    guard = BudgetGuard(max_spend_usd=1.0)
    guard.precheck(0.4)
    guard.record(0.4)
    guard.precheck(0.5)
    guard.record(0.5)  # spent 0.9 of 1.0
    with pytest.raises(BudgetExceededError) as excinfo:
        guard.precheck(0.2)  # would project 1.1 -- must refuse BEFORE the call
    assert "cap" in str(excinfo.value)
    assert guard.spent_usd == pytest.approx(0.9)  # the refused call charged nothing
