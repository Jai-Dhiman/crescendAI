"""Labeled query corpus loader.

Reads candidates.yaml files for each slug, resolves audio from cache,
windows each recording's chroma, and tags each window with piece_id and
is_in_catalog flag. No network calls; missing audio files are excluded and
counted explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from piece_id_eval.query_chroma import audio_to_chroma, window_chroma


@dataclass
class LabeledQueryWindow:
    query_id: str          # "{slug}/{video_id}/{window_idx}"
    slug: str
    video_id: str
    piece_id: str          # catalog piece_id from eval_piece_map.json
    is_in_catalog: bool    # False if slug is in holdout_slugs
    chroma: np.ndarray     # shape (12, window_frames)


@dataclass
class LoadResult:
    windows: list[LabeledQueryWindow]
    excluded_count: int    # recordings skipped (missing audio, not approved, etc.)


class QuerySet:
    @staticmethod
    def load(
        slugs: list[str],
        eval_root: Path,
        piece_map_path: Path,
        audio_cache_root: Path,
        holdout_slugs: list[str],
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
    ) -> LoadResult:
        """Load labeled query windows for the given slugs.

        Args:
            slugs: list of practice_eval slug names (e.g. "bach_prelude_c_wtc1").
            eval_root: path to practice_eval/ directory containing slug subdirs.
            piece_map_path: path to eval_piece_map.json (slug -> catalog piece_id).
            audio_cache_root: root dir where audio/{video_id}.wav files live under
                              each slug subdir.
            holdout_slugs: slugs to tag as is_in_catalog=False.
            window_seconds: chroma window length in seconds.
            hop_seconds: hop between windows in seconds.

        Returns:
            LoadResult with all labeled windows and excluded recording count.

        Raises:
            FileNotFoundError: if piece_map_path does not exist.
            KeyError: if a slug is missing from the piece map.
        """
        if not piece_map_path.exists():
            raise FileNotFoundError(f"eval_piece_map.json not found: {piece_map_path}")
        piece_map: dict[str, str] = json.loads(piece_map_path.read_text())

        windows: list[LabeledQueryWindow] = []
        excluded_count = 0
        holdout_set = set(holdout_slugs)

        for slug in slugs:
            if slug not in piece_map:
                raise KeyError(
                    f"slug {slug!r} not found in eval_piece_map.json at {piece_map_path}"
                )
            piece_id = piece_map[slug]
            is_in_catalog = slug not in holdout_set

            candidates_path = eval_root / slug / "candidates.yaml"
            if not candidates_path.exists():
                excluded_count += 1
                continue

            with open(candidates_path) as f:
                data = yaml.safe_load(f)

            for recording in data.get("recordings", []):
                if not recording.get("approved", False):
                    continue
                video_id = recording["video_id"]
                audio_path = audio_cache_root / slug / "audio" / f"{video_id}.wav"
                if not audio_path.exists():
                    excluded_count += 1
                    continue

                chroma, frame_rate_hz = audio_to_chroma(audio_path)
                chroma_windows = window_chroma(
                    chroma, frame_rate_hz, window_seconds, hop_seconds
                )
                for idx, win in enumerate(chroma_windows):
                    windows.append(LabeledQueryWindow(
                        query_id=f"{slug}/{video_id}/{idx}",
                        slug=slug,
                        video_id=video_id,
                        piece_id=piece_id,
                        is_in_catalog=is_in_catalog,
                        chroma=win,
                    ))

        return LoadResult(windows=windows, excluded_count=excluded_count)
