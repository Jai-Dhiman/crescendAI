"""The not-installed error path -- the only Tinker surface tests touch."""
from __future__ import annotations

import importlib.util

import pytest

_TINKER_MISSING = importlib.util.find_spec("tinker") is None


@pytest.mark.skipif(
    not _TINKER_MISSING,
    reason="tinker SDK installed in this env; the not-installed path is unreachable",
)
def test_missing_sdk_raises_with_install_instructions():
    from audio_teacher.tinker_client import TinkerNotInstalledError, TinkerProbeClient

    with pytest.raises(TinkerNotInstalledError) as excinfo:
        TinkerProbeClient(
            sample_rate=16000,
            usd_per_1m_input_tokens=1.0,
            usd_per_1m_output_tokens=3.0,
        )
    assert "uv add tinker" in str(excinfo.value)
