"""MIDI to audio rendering using FluidSynth or Pianoteq.

Renders MIDI files to WAV audio for subsequent MuQ embedding extraction.
FluidSynth (with Salamander soundfont) is ~12x faster than Pianoteq.
"""

import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from .alignment_dataset import path_to_cache_key


class RenderBackend(Enum):
    FLUIDSYNTH = "fluidsynth"
    PIANOTEQ = "pianoteq"


# Default paths
PIANOTEQ_PATH = Path("/Applications/Pianoteq 9/Pianoteq 9.app/Contents/MacOS/Pianoteq 9")
SALAMANDER_SF2 = Path(__file__).parent.parent.parent.parent / "data" / "soundfonts" / "SalamanderGrandPiano.sf2"

# Default preset for Pianoteq
DEFAULT_PRESET = "NY Steinway D Classical"


def render_midi_fluidsynth(
    midi_path: Path,
    output_path: Path,
    soundfont_path: Path = SALAMANDER_SF2,
    sample_rate: int = 44100,
) -> bool:
    """Render a MIDI file to WAV using FluidSynth.

    Args:
        midi_path: Path to input MIDI file.
        output_path: Path for output WAV file.
        soundfont_path: Path to .sf2 soundfont file.
        sample_rate: Sample rate for output audio.

    Returns:
        True if rendering succeeded, False otherwise.
    """
    midi_path = Path(midi_path)
    output_path = Path(output_path)

    if not soundfont_path.exists():
        raise FileNotFoundError(f"Soundfont not found at: {soundfont_path}")

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "fluidsynth",
        "-ni",  # No shell, non-interactive
        "-g", "1.0",  # Gain
        "-R", "0",  # No reverb (faster)
        "-C", "0",  # No chorus (faster)
        "-r", str(sample_rate),
        "-F", str(output_path),
        str(soundfont_path),
        str(midi_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
        return output_path.exists() and output_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        print(f"Timeout rendering: {midi_path}")
        return False
    except Exception as e:
        print(f"Error rendering {midi_path}: {e}")
        return False


def render_midi_pianoteq(
    midi_path: Path,
    output_path: Path,
    preset: str = DEFAULT_PRESET,
    sample_rate: int = 44100,
    pianoteq_path: Path = PIANOTEQ_PATH,
    normalize: bool = True,
) -> bool:
    """Render a MIDI file to WAV using Pianoteq.

    Args:
        midi_path: Path to input MIDI file.
        output_path: Path for output WAV file.
        preset: Pianoteq preset name.
        sample_rate: Sample rate for output audio.
        pianoteq_path: Path to Pianoteq binary.
        normalize: Whether to normalize output volume.

    Returns:
        True if rendering succeeded, False otherwise.
    """
    midi_path = Path(midi_path)
    output_path = Path(output_path)

    if not pianoteq_path.exists():
        raise FileNotFoundError(f"Pianoteq not found at: {pianoteq_path}")

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(pianoteq_path),
        "--headless",
        "--preset", preset,
        "--midi", str(midi_path),
        "--wav", str(output_path),
        "--rate", str(sample_rate),
    ]

    if normalize:
        cmd.append("--normalize")

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return output_path.exists()
    except subprocess.TimeoutExpired:
        print(f"Timeout rendering: {midi_path}")
        return False
    except Exception as e:
        print(f"Error rendering {midi_path}: {e}")
        return False


def render_midi_to_wav(
    midi_path: Path,
    output_path: Path,
    backend: RenderBackend = RenderBackend.FLUIDSYNTH,
    sample_rate: int = 44100,
    preset: str = DEFAULT_PRESET,
    soundfont_path: Path = SALAMANDER_SF2,
    pianoteq_path: Path = PIANOTEQ_PATH,
) -> bool:
    """Render a MIDI file to WAV.

    Args:
        midi_path: Path to input MIDI file.
        output_path: Path for output WAV file.
        backend: Rendering backend (FLUIDSYNTH or PIANOTEQ).
        sample_rate: Sample rate for output audio.
        preset: Pianoteq preset name (only for Pianoteq backend).
        soundfont_path: Path to .sf2 soundfont (only for FluidSynth backend).
        pianoteq_path: Path to Pianoteq binary (only for Pianoteq backend).

    Returns:
        True if rendering succeeded, False otherwise.
    """
    if backend == RenderBackend.FLUIDSYNTH:
        return render_midi_fluidsynth(midi_path, output_path, soundfont_path, sample_rate)
    else:
        return render_midi_pianoteq(midi_path, output_path, preset, sample_rate, pianoteq_path)


def render_batch(
    midi_files: List[Tuple[Path, Path]],
    backend: RenderBackend = RenderBackend.FLUIDSYNTH,
    sample_rate: int = 44100,
    preset: str = DEFAULT_PRESET,
    soundfont_path: Path = SALAMANDER_SF2,
    pianoteq_path: Path = PIANOTEQ_PATH,
    progress_callback: Optional[callable] = None,
) -> Tuple[int, int]:
    """Render multiple MIDI files to WAV.

    Args:
        midi_files: List of (midi_path, output_path) tuples.
        backend: Rendering backend (FLUIDSYNTH or PIANOTEQ).
        sample_rate: Sample rate for output audio.
        preset: Pianoteq preset name (only for Pianoteq backend).
        soundfont_path: Path to .sf2 soundfont (only for FluidSynth backend).
        pianoteq_path: Path to Pianoteq binary (only for Pianoteq backend).
        progress_callback: Optional callback(completed, total) for progress updates.

    Returns:
        Tuple of (successful_count, failed_count).
    """
    from tqdm.auto import tqdm

    successful = 0
    failed = 0
    total = len(midi_files)

    # Filter to only files that need rendering
    to_render = [(m, o) for m, o in midi_files if not o.exists()]
    already_done = total - len(to_render)
    successful += already_done

    if not to_render:
        print(f"All {total} files already rendered.")
        return successful, failed

    print(f"Rendering {len(to_render)} files with {backend.value}...")

    for midi_path, output_path in tqdm(to_render, desc=f"Rendering ({backend.value})"):
        success = render_midi_to_wav(
            midi_path,
            output_path,
            backend=backend,
            sample_rate=sample_rate,
            preset=preset,
            soundfont_path=soundfont_path,
            pianoteq_path=pianoteq_path,
        )

        if success:
            successful += 1
        else:
            failed += 1

        if progress_callback:
            progress_callback(successful + failed, total)

    return successful, failed


def get_render_jobs_for_asap(
    performances,
    asap_root: Path,
    audio_cache_dir: Path,
    render_scores: bool = True,
    render_performances: bool = True,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Get list of MIDI files to render for ASAP dataset.

    Args:
        performances: List of ASAPPerformance objects.
        asap_root: Root directory of ASAP dataset.
        audio_cache_dir: Directory to store rendered audio.
        render_scores: Whether to render score MIDI files.
        render_performances: Whether to render performance MIDI files.

    Returns:
        Tuple of (score_jobs, performance_jobs) where each job is (midi_path, wav_path).
    """
    asap_root = Path(asap_root)
    audio_cache_dir = Path(audio_cache_dir)

    score_jobs = []
    perf_jobs = []

    # Track unique scores to avoid duplicates
    seen_scores = set()

    for perf in performances:
        # Performance MIDI
        if render_performances and perf.midi_performance_path:
            midi_path = asap_root / perf.midi_performance_path
            # Use consistent cache key (matches alignment_dataset)
            safe_name = path_to_cache_key(perf.performance_id)
            wav_path = audio_cache_dir / "performances" / f"{safe_name}.wav"

            if midi_path.exists():
                perf_jobs.append((midi_path, wav_path))

        # Score MIDI (deduplicate by path)
        if render_scores and perf.midi_score_path:
            score_key = str(perf.midi_score_path)
            if score_key not in seen_scores:
                seen_scores.add(score_key)
                midi_path = asap_root / perf.midi_score_path
                # Use consistent cache key (matches alignment_dataset)
                safe_name = path_to_cache_key(score_key)
                wav_path = audio_cache_dir / "scores" / f"{safe_name}.wav"

                if midi_path.exists():
                    score_jobs.append((midi_path, wav_path))

    return score_jobs, perf_jobs
