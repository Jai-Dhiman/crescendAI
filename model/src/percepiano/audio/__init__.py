"""
Audio processing module for PercePiano.

Provides MIDI-to-audio rendering and MERT feature extraction.
"""

from .render_midi import (
    render_midi_to_wav,
    batch_render_midi,
    download_salamander_soundfont,
    check_fluidsynth_installed,
    get_audio_duration,
    SOUNDFONT_URL,
)
from .extract_mert import (
    MERT330MExtractor,
    batch_extract_mert,
    get_embedding_stats,
)

__all__ = [
    # Rendering
    "render_midi_to_wav",
    "batch_render_midi",
    "download_salamander_soundfont",
    "check_fluidsynth_installed",
    "get_audio_duration",
    "SOUNDFONT_URL",
    # MERT extraction
    "MERT330MExtractor",
    "batch_extract_mert",
    "get_embedding_stats",
]
