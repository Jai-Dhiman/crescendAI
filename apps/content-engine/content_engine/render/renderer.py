"""Renderer: composes 9:16 video from clip audio + voiceover + overlays."""
from __future__ import annotations
import subprocess
from pathlib import Path
from content_engine.pipeline.episode import Episode
from content_engine.render.templates import CtaTemplate


class RendererError(Exception):
    pass


class Renderer:
    def __init__(self, output_dir: Path, clip_paths: dict[str, Path]):
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._clip_paths = {k: Path(v) for k, v in clip_paths.items()}

    def render(self, episode: Episode, cta_template: CtaTemplate) -> Path:
        if episode.id not in self._clip_paths:
            raise RendererError(f"no clip path registered for episode {episode.id}")
        if episode.voiceover_path is None:
            raise RendererError(f"episode {episode.id} has no voiceover_path")

        clip = self._clip_paths[episode.id]
        voiceover = Path(episode.voiceover_path)
        out = self._out / f"{episode.id}.mp4"

        if cta_template.watermark_enabled:
            # drawbox acts as a watermark placeholder (drawtext requires libfreetype)
            watermark_filter = "drawbox=x=30:y=30:w=200:h=50:color=white@0.6:t=fill"
        else:
            watermark_filter = "null"

        filter_complex = (
            f"color=c=black:s=1080x1920:d=10[bg];"
            f"[1:a]volume=1.0[vo];"
            f"[0:a][vo]amix=inputs=2:duration=longest[a];"
            f"[bg]{watermark_filter}[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-fflags", "+bitexact",
            "-i", str(clip),
            "-i", str(voiceover),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            "-g", "1",
            "-c:a", "aac", "-b:a", "128k",
            "-flags:v", "+bitexact",
            "-flags:a", "+bitexact",
            "-fflags", "+bitexact",
            "-map_metadata", "-1",
            "-movflags", "+faststart+empty_moov",
            "-shortest",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RendererError(f"ffmpeg exit {result.returncode}: {result.stderr[-500:]}")
        if not out.exists():
            raise RendererError(f"renderer did not produce expected output: {out}")
        return out
