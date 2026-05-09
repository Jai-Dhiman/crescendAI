"""Verifies Renderer produces a valid 9:16 mp4 with correct duration and audio."""
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.render.renderer import Renderer
from content_engine.render.templates import CtaTemplate


def _make_silent_wav(path: Path, duration_sec: float = 3.0):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
         "-t", str(duration_sec), str(path)],
        check=True, capture_output=True,
    )


def _ffprobe(path: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(res.stdout)


def test_render_produces_9_16_mp4_with_audio(tmp_path):
    clip = tmp_path / "clip.wav"
    voiceover = tmp_path / "vo.wav"
    _make_silent_wav(clip, 3.0)
    _make_silent_wav(voiceover, 3.0)

    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    ep = Episode(
        id="ep_render_test",
        candidate_url="x",
        source_type="youtube_amateur",
        state=State.RECORDED,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
        observation={"dimension": "phrasing", "time_range": [0.5, 2.0], "plain_english": "rushed"},
        script_text="Hook. Observation. End.",
        voiceover_path=str(voiceover),
    )
    out_dir = tmp_path / "renders"
    out_dir.mkdir()
    r = Renderer(output_dir=out_dir, clip_paths={ep.id: clip})

    out = r.render(ep, CtaTemplate.for_phase("A"))

    assert out.exists()
    probe = _ffprobe(out)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert video_stream["width"] == 1080
    assert video_stream["height"] == 1920
    assert float(probe["format"]["duration"]) > 2.0
    assert audio_stream is not None
