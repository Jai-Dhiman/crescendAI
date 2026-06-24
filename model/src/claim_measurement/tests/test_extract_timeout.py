"""Per-clip timeout guard for the clean-bundle extractor (#98).

parangonar alignment can blow up combinatorially on some real transcriptions
(the arpeggiated prelude hung >1h in #95). The extraction runner wraps each clip
in a wall-clock limit so one pathological piece is recorded as 'timeout' instead
of blocking the whole corpus run.
"""
from __future__ import annotations

import time

import pytest

from claim_measurement.extract_cli import _time_limit


def test_time_limit_raises_on_overrun():
    with pytest.raises(TimeoutError):
        with _time_limit(1):
            time.sleep(3)


def test_time_limit_passes_under_budget():
    with _time_limit(5):
        result = sum(range(100))
    assert result == 4950


def test_time_limit_zero_disables_guard():
    # 0 (or negative) means no limit; the body runs without arming an alarm.
    with _time_limit(0):
        result = 1 + 1
    assert result == 2
