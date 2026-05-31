import json
from pathlib import Path

import numpy as np
import pytest

from chroma_dtw_eval.dtw_runner import DtwRunnerError, run_dtw


SCORE_JSON = [{
    "bar_number": 1, "start_tick": 0, "start_seconds": 0.0,
    "time_signature": "4/4",
    "notes": [{
        "pitch": 60, "pitch_name": "C4", "velocity": 80,
        "onset_tick": 0, "onset_seconds": 0.0,
        "duration_ticks": 480, "duration_seconds": 0.2, "track": 0,
    }],
    "pedal_events": [], "note_count": 1,
    "pitch_range": [60], "mean_velocity": 80,
}]


def test_run_dtw_returns_result_on_valid_input(tmp_path):
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(SCORE_JSON))
    chroma = np.zeros((12, 2), dtype=np.float32)
    chroma[0, 0] = 1.0
    chroma[0, 1] = 1.0
    result = run_dtw(chroma, score_path, frame_rate_hz=10.0, decim_hz=10.0)
    assert isinstance(result.predicted_score_frame, int)
    assert isinstance(result.cost, float)
    assert result.bar_per_frame and all(isinstance(b, int) for b in result.bar_per_frame)


def test_run_dtw_raises_on_missing_score(tmp_path):
    chroma = np.zeros((12, 2), dtype=np.float32)
    with pytest.raises(DtwRunnerError):
        run_dtw(chroma, tmp_path / "does_not_exist.json", frame_rate_hz=10.0, decim_hz=10.0)
