"""Verifies Renderer is deterministic: identical inputs -> byte-identical output."""
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.render.renderer import Renderer
from content_engine.render.templates import CtaTemplate


def _make_silent_wav(path: Path, duration_sec: float = 3.0):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-t", str(duration_sec), str(path)],
        check=True, capture_output=True,
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_render_is_deterministic(tmp_path):
    clip = tmp_path / "clip.wav"
    vo = tmp_path / "vo.wav"
    _make_silent_wav(clip, 3.0)
    _make_silent_wav(vo, 3.0)

    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    ep = Episode(
        id="ep_det",
        candidate_url="x",
        source_type="youtube_amateur",
        state=State.RECORDED,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
        observation={"dimension": "phrasing", "time_range": [0.5, 2.0], "plain_english": "x"},
        script_text="x",
        voiceover_path=str(vo),
    )

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    r_a = Renderer(output_dir=out_a, clip_paths={ep.id: clip})
    r_b = Renderer(output_dir=out_b, clip_paths={ep.id: clip})

    f_a = r_a.render(ep, CtaTemplate.for_phase("A"))
    f_b = r_b.render(ep, CtaTemplate.for_phase("A"))

    assert _hash(f_a) == _hash(f_b)
