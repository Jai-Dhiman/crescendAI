"""
MIDI to Audio Rendering using FluidSynth.

Renders MIDI files to WAV audio using FluidSynth with the Salamander Grand Piano
soundfont for high-quality piano synthesis.
"""

import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm.auto import tqdm

# Salamander C5 Lite soundfont - Google Drive file ID
# Source: https://sites.google.com/view/hed-sounds/salamander-c5-light
# 7 velocity layers, 44.1kHz/16-bit, ~24MB
SOUNDFONT_GDRIVE_ID = "0B5gPxvwx-I4KWjZ2SHZOLU42dHM"


def check_fluidsynth_installed() -> bool:
    """Check if FluidSynth is installed and available."""
    result = subprocess.run(["which", "fluidsynth"], capture_output=True)
    return result.returncode == 0


def download_salamander_soundfont(
    output_path: Path,
    gdrive_id: str = SOUNDFONT_GDRIVE_ID,
) -> Path:
    """
    Download and extract Salamander Grand Piano soundfont from Google Drive.

    Args:
        output_path: Path where the .sf2 file should be saved
        gdrive_id: Google Drive file ID

    Returns:
        Path to the extracted .sf2 file

    Raises:
        RuntimeError: If download or extraction fails
    """
    import gdown

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Soundfont already exists: {output_path}")
        return output_path

    print("Downloading Salamander C5 Lite soundfont (~25MB)...")
    zip_path = output_path.parent / "salamander.zip"

    # Download from Google Drive
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, str(zip_path), quiet=False)

    if not zip_path.exists():
        raise RuntimeError("Download failed - file not created")

    print(f"Downloaded to {zip_path}")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_path.parent)

    # Find the .sf2 file
    sf2_files = list(output_path.parent.rglob("*.sf2"))
    if not sf2_files:
        raise RuntimeError("No .sf2 file found in archive")

    # Move to expected location
    sf2_files[0].rename(output_path)
    print(f"Soundfont ready: {output_path}")

    # Cleanup
    zip_path.unlink(missing_ok=True)

    # Clean up extracted directory
    for item in output_path.parent.iterdir():
        if item.is_dir() and "Salamander" in item.name:
            import shutil

            shutil.rmtree(item)

    return output_path


def render_midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    soundfont_path: Path,
    sample_rate: int = 44100,
    gain: float = 0.8,
    timeout: int = 60,
) -> bool:
    """
    Render a MIDI file to WAV using FluidSynth.

    Args:
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
        soundfont_path: Path to .sf2 soundfont
        sample_rate: Output sample rate (default 44100)
        gain: Output gain to avoid clipping (default 0.8)
        timeout: Timeout in seconds per file (default 60)

    Returns:
        True if successful, False otherwise
    """
    try:
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "fluidsynth",
                "-ni",  # Non-interactive
                "-a", "file",  # Use file audio driver (no playback)
                "-F", str(wav_path),  # Output file
                "-r", str(sample_rate),  # Sample rate
                "-g", str(gain),  # Gain
                str(soundfont_path),
                str(midi_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            print(f"Error rendering {midi_path.name}: {result.stderr}")
            return False

        return wav_path.exists()

    except subprocess.TimeoutExpired:
        print(f"Timeout rendering {midi_path.name}")
        return False
    except Exception as e:
        print(f"Exception rendering {midi_path.name}: {e}")
        return False


def batch_render_midi(
    midi_dir: Path,
    output_dir: Path,
    soundfont_path: Path,
    label_keys: Optional[List[str]] = None,
    max_workers: int = 4,
    skip_existing: bool = True,
    sample_rate: int = 44100,
    gain: float = 0.8,
) -> Tuple[int, int]:
    """
    Batch render MIDI files to WAV.

    Args:
        midi_dir: Directory containing MIDI files
        output_dir: Directory for output WAV files
        soundfont_path: Path to soundfont
        label_keys: List of segment keys to render (if None, renders all .mid files)
        max_workers: Number of parallel workers
        skip_existing: Skip files that already exist
        sample_rate: Output sample rate
        gain: Output gain

    Returns:
        Tuple of (successful, failed) counts
    """
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    soundfont_path = Path(soundfont_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of files to render
    to_render = []

    if label_keys is not None:
        # Render specific files based on label keys
        for key in label_keys:
            midi_path = midi_dir / f"{key}.mid"
            wav_path = output_dir / f"{key}.wav"

            if skip_existing and wav_path.exists():
                continue

            if midi_path.exists():
                to_render.append((midi_path, wav_path))
            else:
                print(f"MIDI not found: {key}")
    else:
        # Render all MIDI files in directory
        for midi_path in midi_dir.glob("*.mid"):
            wav_path = output_dir / f"{midi_path.stem}.wav"

            if skip_existing and wav_path.exists():
                continue

            to_render.append((midi_path, wav_path))

    total_expected = len(label_keys) if label_keys else len(list(midi_dir.glob("*.mid")))
    already_rendered = total_expected - len(to_render)

    print(f"Files to render: {len(to_render)}")
    print(f"Already rendered: {already_rendered}")

    if not to_render:
        return total_expected, 0

    # Render in parallel
    successful = already_rendered
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                render_midi_to_wav,
                midi_path,
                wav_path,
                soundfont_path,
                sample_rate,
                gain,
            ): midi_path.stem
            for midi_path, wav_path in to_render
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
            if future.result():
                successful += 1
            else:
                failed += 1

    return successful, failed


def get_audio_duration(wav_path: Path) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        wav_path: Path to WAV file

    Returns:
        Duration in seconds
    """
    import wave

    with wave.open(str(wav_path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / float(rate)
