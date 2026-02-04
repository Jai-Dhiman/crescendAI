"""Pianoteq MIDI-to-audio rendering with caching.

Pianoteq is a physically-modeled piano synthesizer that can render
MIDI files to high-quality audio.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import List, Optional

from ..config import MUQ_SAMPLE_RATE


class PianoteqRenderError(Exception):
    """Exception raised when Pianoteq rendering fails."""

    pass


class PianoteqRenderer:
    """MIDI-to-audio renderer using Pianoteq with caching.

    Renders MIDI files to WAV audio using Pianoteq's command-line interface.
    Implements content-based caching to avoid re-rendering unchanged files.
    """

    def __init__(
        self,
        executable: str,
        preset: str = "D4 Classical",
        cache_dir: Optional[Path] = None,
        sample_rate: int = MUQ_SAMPLE_RATE,
    ):
        """Initialize Pianoteq renderer.

        Args:
            executable: Path to Pianoteq executable (e.g., /usr/local/bin/pianoteq).
            preset: Pianoteq preset name to use for rendering.
            cache_dir: Directory for caching rendered audio. If None, no caching.
            sample_rate: Output sample rate (default: 24000 Hz for MuQ).

        Raises:
            FileNotFoundError: If executable doesn't exist.
        """
        self.executable = Path(executable)
        if not self.executable.exists():
            raise FileNotFoundError(f"Pianoteq executable not found: {executable}")

        self.preset = preset
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sample_rate = sample_rate

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_index_path = self.cache_dir / ".cache_index.json"
            self._load_cache_index()
        else:
            self._cache_index = {}

    def _load_cache_index(self):
        """Load cache index from disk."""
        if self._cache_index_path.exists():
            with open(self._cache_index_path) as f:
                self._cache_index = json.load(f)
        else:
            self._cache_index = {}

    def _save_cache_index(self):
        """Save cache index to disk."""
        if self.cache_dir:
            with open(self._cache_index_path, "w") as f:
                json.dump(self._cache_index, f, indent=2)

    def _compute_cache_key(self, midi_path: Path) -> str:
        """Compute cache key from MIDI content and preset.

        Uses SHA256 of MIDI file content combined with preset name
        to ensure cache invalidation when either changes.
        """
        midi_path = Path(midi_path)

        # Hash MIDI content
        with open(midi_path, "rb") as f:
            midi_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        # Hash preset name
        preset_hash = hashlib.sha256(self.preset.encode()).hexdigest()[:8]

        # Combine with sample rate
        return f"{midi_path.stem}_{midi_hash}_{preset_hash}_{self.sample_rate}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        if not self.cache_dir:
            raise ValueError("Cache directory not set")
        return self.cache_dir / f"{cache_key}.wav"

    def _is_cached(self, cache_key: str) -> bool:
        """Check if a render is cached."""
        if not self.cache_dir:
            return False

        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists() and cache_key in self._cache_index

    def render(
        self,
        midi_path: Path,
        output_path: Optional[Path] = None,
        use_cache: bool = True,
    ) -> Path:
        """Render a MIDI file to WAV audio.

        Args:
            midi_path: Path to input MIDI file.
            output_path: Path for output WAV file. If None, uses cache directory
                        or same directory as MIDI with .wav extension.
            use_cache: Whether to use cached renders if available.

        Returns:
            Path to the rendered WAV file.

        Raises:
            PianoteqRenderError: If rendering fails.
            FileNotFoundError: If MIDI file doesn't exist.
        """
        midi_path = Path(midi_path)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        # Compute cache key
        cache_key = self._compute_cache_key(midi_path)

        # Check cache
        if use_cache and self._is_cached(cache_key):
            cached_path = self._get_cache_path(cache_key)
            if output_path and output_path != cached_path:
                # Copy from cache to requested output
                import shutil

                shutil.copy(cached_path, output_path)
                return output_path
            return cached_path

        # Determine output path
        if output_path is None:
            if self.cache_dir:
                output_path = self._get_cache_path(cache_key)
            else:
                output_path = midi_path.with_suffix(".wav")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build Pianoteq command
        cmd = [
            str(self.executable),
            "--headless",
            "--preset", self.preset,
            "--rate", str(self.sample_rate),
            "--midi", str(midi_path),
            "--wav", str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise PianoteqRenderError(
                    f"Pianoteq failed with code {result.returncode}: {result.stderr}"
                )

            if not output_path.exists():
                raise PianoteqRenderError(
                    f"Pianoteq completed but output file not found: {output_path}"
                )

        except subprocess.TimeoutExpired as e:
            raise PianoteqRenderError(f"Pianoteq rendering timed out: {e}") from e
        except FileNotFoundError as e:
            raise PianoteqRenderError(f"Failed to execute Pianoteq: {e}") from e

        # Update cache index
        if use_cache and self.cache_dir:
            self._cache_index[cache_key] = {
                "midi_path": str(midi_path),
                "preset": self.preset,
                "sample_rate": self.sample_rate,
            }
            self._save_cache_index()

        return output_path

    def render_batch(
        self,
        midi_paths: List[Path],
        output_dir: Optional[Path] = None,
        use_cache: bool = True,
    ) -> List[Path]:
        """Render multiple MIDI files to audio.

        Args:
            midi_paths: List of paths to MIDI files.
            output_dir: Directory for output WAV files. If None, uses cache.
            use_cache: Whether to use cached renders.

        Returns:
            List of paths to rendered WAV files.

        Raises:
            PianoteqRenderError: If any rendering fails.
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for midi_path in midi_paths:
            midi_path = Path(midi_path)

            if output_dir:
                output_path = output_dir / f"{midi_path.stem}.wav"
            else:
                output_path = None

            wav_path = self.render(midi_path, output_path, use_cache)
            results.append(wav_path)

        return results

    def clear_cache(self):
        """Clear the render cache."""
        if self.cache_dir:
            for wav_file in self.cache_dir.glob("*.wav"):
                wav_file.unlink()
            self._cache_index = {}
            self._save_cache_index()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if not self.cache_dir:
            return {"cached_renders": 0, "cache_size_mb": 0}

        wav_files = list(self.cache_dir.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in wav_files)

        return {
            "cached_renders": len(wav_files),
            "cache_size_mb": total_size / (1024 * 1024),
        }
