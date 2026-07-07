"""Verify the follower_bench package is importable (editable-install wiring)."""
from __future__ import annotations

import follower_bench


def test_package_is_importable() -> None:
    assert follower_bench is not None
