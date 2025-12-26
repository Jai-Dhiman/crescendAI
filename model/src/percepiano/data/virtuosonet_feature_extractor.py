"""
VirtuosoNet Feature Extractor for PercePiano Replica.

This module wraps VirtuosoNet's pyScoreParser to extract 84 features
matching the original PercePiano implementation exactly. The features include:
- 79 base features (score-level information: pitch, duration, dynamics, tempo, articulation)
- 5 preserved unnormalized features (midi_pitch_unnorm, duration_unnorm, etc.)

The unnormalized features are critical for:
- Key augmentation (midi_pitch_unnorm provides raw MIDI pitch 21-108)
- Preserving original scale information before z-score normalization

Feature order matches original PercePiano data_for_training.py VNET_INPUT_KEYS exactly.

Reference:
- VirtuosoNet: https://github.com/jdasam/virtuosoNet
- PercePiano: https://github.com/JonghoKimSNU/PercePiano
"""

import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add VirtuosoNet to path - need both the parent (for package imports) and pyScoreParser itself
VIRTUOSO_ROOT = (
    Path(__file__).parent.parent.parent
    / "data"
    / "raw"
    / "PercePiano"
    / "virtuoso"
    / "virtuoso"
)
VIRTUOSO_PATH = VIRTUOSO_ROOT / "pyScoreParser"

# Add parent for package-style imports (from pyScoreParser import ...)
if str(VIRTUOSO_ROOT) not in sys.path:
    sys.path.insert(0, str(VIRTUOSO_ROOT))
# Add pyScoreParser directly for direct imports (import feature_extraction)
if str(VIRTUOSO_PATH) not in sys.path:
    sys.path.insert(0, str(VIRTUOSO_PATH))


# VirtuosoNet Input Feature Keys (from data_for_training.py)
# Matches original PercePiano exactly: 79 base features
VNET_INPUT_KEYS = (
    "midi_pitch",           # 0
    "duration",             # 1
    "beat_importance",      # 2
    "measure_length",       # 3
    "qpm_primo",            # 4
    "section_tempo",        # 5 - restored to match original
    "following_rest",       # 6
    "distance_from_abs_dynamic",   # 7
    "distance_from_recent_tempo",  # 8
    "beat_position",        # 9
    "xml_position",         # 10
    "grace_order",          # 11
    "preceded_by_grace_note",      # 12
    "followed_by_fermata_rest",    # 13
    "pitch",                # 14-26 (13-dim)
    "tempo",                # 27-31 (5-dim)
    "dynamic",              # 32-35 (4-dim)
    "time_sig_vec",         # 36-44 (9-dim)
    "slur_beam_vec",        # 45-50 (6-dim)
    "composer_vec",         # 51-67 (17-dim)
    "notation",             # 68-76 (9-dim)
    "tempo_primo",          # 77-78 (2-dim)
)

# Features that need z-score normalization (9 scalar features)
# Matches original PercePiano data_for_training.py exactly
NORM_FEAT_KEYS = (
    "midi_pitch",
    "duration",
    "beat_importance",
    "measure_length",
    "qpm_primo",
    "section_tempo",  # restored to match original
    "following_rest",
    "distance_from_abs_dynamic",
    "distance_from_recent_tempo",
)

# Features to preserve BEFORE normalization (original PercePiano data_for_training.py:21)
# These are appended as _unnorm variants after the base 79 features
PRESERVE_FEAT_KEYS = (
    "midi_pitch",
    "duration",
    "beat_importance",
    "measure_length",
    "following_rest",
)

# Feature dimensions (based on VirtuosoNet implementation)
# Matches original PercePiano: 79 base features (14 scalar + 65 vector)
FEATURE_DIMS = {
    "midi_pitch": 1,
    "duration": 1,
    "beat_importance": 1,
    "measure_length": 1,
    "qpm_primo": 1,
    "section_tempo": 1,  # restored to match original
    "following_rest": 1,
    "distance_from_abs_dynamic": 1,
    "distance_from_recent_tempo": 1,
    "beat_position": 1,
    "xml_position": 1,
    "grace_order": 1,
    "preceded_by_grace_note": 1,
    "followed_by_fermata_rest": 1,
    "pitch": 13,  # octave (normalized) + 12-class one-hot
    "tempo": 5,  # tempo marking embedding
    "dynamic": 4,  # dynamic marking embedding
    "time_sig_vec": 9,  # 5 numerator + 4 denominator
    "slur_beam_vec": 6,  # slur start/continue/stop + beam start/continue/stop
    "composer_vec": 17,  # 16 composers + unknown
    "notation": 9,  # trill, tenuto, accent, staccato, fermata, arpeggiate, strong_accent, cue, slash
    "tempo_primo": 2,  # initial tempo embedding
}

BASE_FEATURE_DIM = sum(
    FEATURE_DIMS.values()
)  # = 79 (14 scalar + 65 vector features) - matches original PercePiano
NUM_PRESERVE_FEATURES = len(PRESERVE_FEAT_KEYS)  # = 5 unnorm features
TOTAL_FEATURE_DIM = (
    BASE_FEATURE_DIM + NUM_PRESERVE_FEATURES
)  # = 84 (79 base + 5 unnorm)

# Feature indices for key augmentation (after unnorm features are appended)
# Matches original PercePiano feature layout
MIDI_PITCH_IDX = 0  # Normalized midi_pitch (z-score)
MIDI_PITCH_UNNORM_IDX = (
    BASE_FEATURE_DIM  # Raw midi_pitch (21-108), appended at end = 79
)
PITCH_VEC_START = 14  # pitch vector starts at index 14 (after 14 scalar features)
PITCH_CLASS_START = 15  # octave at 14, pitch class one-hot starts at 15
PITCH_CLASS_END = 27  # 12 pitch classes (indices 15-26)


@dataclass
class FeatureStats:
    """Statistics for z-score normalization."""

    mean: Dict[str, float]
    std: Dict[str, float]


class VirtuosoNetFeatureExtractor:
    """
    Extract VirtuosoNet features from aligned MusicXML scores and MIDI performances.

    This class wraps VirtuosoNet's feature extraction pipeline to produce the
    84-dimensional feature vectors matching the original PercePiano implementation:
    - 79 base features (normalized where applicable)
    - 5 preserved unnormalized features (midi_pitch_unnorm, duration_unnorm, etc.)
    """

    def __init__(self, composer: str = "unknown"):
        """
        Initialize the feature extractor.

        Args:
            composer: Composer name for composer_vec encoding.
                     One of: Bach, Balakirev, Beethoven, Brahms, Chopin, Debussy,
                     Glinka, Haydn, Liszt, Mozart, Prokofiev, Rachmaninoff, Ravel,
                     Schubert, Schumann, Scriabin, or "unknown"
        """
        self.composer = composer
        self._import_virtuoso_modules()

    def _import_virtuoso_modules(self):
        """Import VirtuosoNet modules with error handling."""
        try:
            import data_class
            import feature_extraction as feat_ext
            import feature_utils
            import xml_direction_encoding as dir_enc
            import xml_midi_matching as matching

            self.feat_ext = feat_ext
            self.feature_utils = feature_utils
            self.dir_enc = dir_enc
            self.data_class = data_class
            self.matching = matching
            self._virtuoso_available = True
        except ImportError as e:
            print(f"Warning: Could not import VirtuosoNet modules: {e}")
            print(
                f"Make sure {VIRTUOSO_PATH} exists and contains the required modules."
            )
            self._virtuoso_available = False

    def _get_or_create_score_midi(self, score_xml_path: Path) -> Path:
        """
        Get or create a score MIDI file for the given MusicXML.

        The PercePiano API requires a score MIDI (synthesized from MusicXML)
        for score-to-MIDI alignment. This method either finds an existing
        score MIDI or generates one from the XML.

        Args:
            score_xml_path: Path to MusicXML score file

        Returns:
            Path to the score MIDI file
        """
        # Common naming conventions for score MIDIs
        possible_paths = [
            score_xml_path.with_suffix(".mid"),
            score_xml_path.parent / "midi_cleaned.mid",
            score_xml_path.parent / "midi_score.mid",
            score_xml_path.parent / (score_xml_path.stem + "_score.mid"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # No existing score MIDI found - will be generated by ScoreData
        # ScoreData.make_score_midi() creates it if missing
        generated_path = score_xml_path.with_suffix(".mid")
        return generated_path

    def extract_features(
        self,
        score_xml_path: Path,
        performance_midi_path: Path,
    ) -> Dict[str, Any]:
        """
        Extract 79 base VirtuosoNet features from a score-performance pair.

        Note: This returns 79 base features (matching original PercePiano). The
        preprocessing script adds 5 unnorm features to make 84 total.

        Args:
            score_xml_path: Path to MusicXML score file
            performance_midi_path: Path to performance MIDI file

        Returns:
            Dictionary containing:
            - 'input': numpy array of shape (num_notes, 79) - base features
            - 'note_location': dict with 'beat', 'measure', 'voice' arrays
            - 'num_notes': number of notes in the piece
            - 'align_matched': boolean array indicating matched notes
        """
        if not self._virtuoso_available:
            raise RuntimeError("VirtuosoNet modules not available")

        score_xml_path = Path(score_xml_path)
        performance_midi_path = Path(performance_midi_path)

        # Generate or find score MIDI (synthesized from MusicXML)
        # The PercePiano API requires a score MIDI for alignment
        score_midi_path = self._get_or_create_score_midi(score_xml_path)

        # Create ScoreData which loads XML, generates score MIDI if needed,
        # and creates the XML-to-MIDI matching
        try:
            score_data = self.data_class.ScoreData(
                xml_path=str(score_xml_path),
                score_midi_path=str(score_midi_path),
                composer=self.composer,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load score data from {score_xml_path}: {e}")

        # Create score extractor and extract features
        # Note: ScoreExtractor expects a piece_data with score_features dict
        # Add note_location to the feature keys (it's needed for output but not in input vector)
        score_data.score_features = {}
        feature_keys_with_location = list(VNET_INPUT_KEYS) + ["note_location"]
        score_extractor = self.feat_ext.ScoreExtractor(feature_keys_with_location)
        score_features = score_extractor.extract_score_features(score_data)

        # Get note locations
        note_locations = score_features.get("note_location", {})

        # Build the 79-dim base feature vector for each note (matches original PercePiano)
        # (unnorm features are added in preprocessing, making 84 total)
        num_notes = len(score_features.get("midi_pitch", []))
        input_features = np.zeros((num_notes, BASE_FEATURE_DIM), dtype=np.float32)

        feature_idx = 0
        for key in VNET_INPUT_KEYS:
            dim = FEATURE_DIMS[key]
            feat_values = score_features.get(key)

            if feat_values is None:
                feature_idx += dim
                continue

            # Handle scalar vs vector features
            if dim == 1:
                # Scalar feature
                if isinstance(feat_values, (int, float)):
                    # Global scalar (like qpm_primo)
                    input_features[:, feature_idx] = feat_values
                else:
                    # Per-note scalar
                    input_features[:, feature_idx] = np.array(feat_values)
            else:
                # Vector feature
                if isinstance(feat_values[0], (list, tuple, np.ndarray)):
                    # Per-note vector
                    for i, vec in enumerate(feat_values):
                        input_features[i, feature_idx : feature_idx + dim] = vec[:dim]
                else:
                    # Global vector (like tempo_primo, composer_vec)
                    vec = np.array(feat_values[:dim])
                    input_features[:, feature_idx : feature_idx + dim] = vec

            feature_idx += dim

        # Get alignment info
        align_matched = np.ones(num_notes, dtype=bool)  # Default to all matched
        if "align_matched" in score_features:
            align_matched = np.array(score_features["align_matched"], dtype=bool)

        return {
            "input": input_features,
            "note_location": note_locations,
            "num_notes": num_notes,
            "align_matched": align_matched,
            "score_path": str(score_xml_path),
            "perform_path": str(performance_midi_path),
        }

    @staticmethod
    def compute_normalization_stats(
        features_list: List[Dict[str, Any]],
    ) -> FeatureStats:
        """
        Compute z-score normalization statistics from a list of feature dicts.

        Args:
            features_list: List of feature dictionaries from extract_features()

        Returns:
            FeatureStats with mean and std for each normalizable feature
        """
        # Concatenate all feature vectors
        all_features = np.concatenate([f["input"] for f in features_list], axis=0)

        # Compute mean and std for each feature dimension
        means = {}
        stds = {}

        feature_idx = 0
        for key in VNET_INPUT_KEYS:
            dim = FEATURE_DIMS[key]

            if key in NORM_FEAT_KEYS:
                feat_slice = all_features[:, feature_idx : feature_idx + dim]
                means[key] = float(np.mean(feat_slice))
                std = float(np.std(feat_slice))
                stds[key] = std if std > 0 else 1.0  # Avoid division by zero
            else:
                means[key] = 0.0
                stds[key] = 1.0  # No normalization for these

            feature_idx += dim

        return FeatureStats(mean=means, std=stds)

    @staticmethod
    def apply_normalization(
        features: Dict[str, Any], stats: FeatureStats
    ) -> Dict[str, Any]:
        """
        Apply z-score normalization to features.

        Args:
            features: Feature dictionary from extract_features()
            stats: Normalization statistics from compute_normalization_stats()

        Returns:
            Features with normalized input array
        """
        normalized = features.copy()
        input_features = features["input"].copy()

        feature_idx = 0
        for key in VNET_INPUT_KEYS:
            dim = FEATURE_DIMS[key]

            if key in NORM_FEAT_KEYS:
                mean = stats.mean[key]
                std = stats.std[key]
                input_features[:, feature_idx : feature_idx + dim] = (
                    input_features[:, feature_idx : feature_idx + dim] - mean
                ) / std

            feature_idx += dim

        normalized["input"] = input_features
        return normalized


def extract_features_standalone(
    score_xml_path: Path, performance_midi_path: Path, composer: str = "unknown"
) -> Optional[Dict[str, Any]]:
    """
    Standalone function to extract VirtuosoNet features.

    This is a convenience wrapper that handles errors gracefully.

    Args:
        score_xml_path: Path to MusicXML score
        performance_midi_path: Path to performance MIDI
        composer: Composer name

    Returns:
        Feature dictionary or None if extraction fails
    """
    try:
        extractor = VirtuosoNetFeatureExtractor(composer=composer)
        return extractor.extract_features(score_xml_path, performance_midi_path)
    except Exception as e:
        print(f"Feature extraction failed for {performance_midi_path}: {e}")
        return None


def cal_beat_importance(beat_position: float, numerator: int) -> float:
    """
    Calculate beat importance from beat position and time signature numerator.

    This is a standalone implementation that doesn't require VirtuosoNet.

    Args:
        beat_position: Position within measure [0, 1)
        numerator: Time signature numerator

    Returns:
        Beat importance value (0-4 scale)
    """
    if beat_position == 0:
        return 4.0
    elif beat_position == 0.5 and numerator in [2, 4, 6, 12]:
        return 3.0
    elif abs(beat_position - (1 / 3)) < 0.001 and numerator in [3, 9]:
        return 2.0
    elif (beat_position * 4) % 1 == 0 and numerator in [2, 4]:
        return 1.0
    elif (beat_position * 5) % 1 == 0 and numerator in [5]:
        return 2.0
    elif (beat_position * 6) % 1 == 0 and numerator in [3, 6, 12]:
        return 1.0
    elif (beat_position * 8) % 1 == 0 and numerator in [2, 4]:
        return 0.5
    elif (beat_position * 9) % 1 == 0 and numerator in [9]:
        return 1.0
    elif (beat_position * 12) % 1 == 0 and numerator in [3, 6, 12]:
        return 0.5
    elif numerator == 7:
        if abs((beat_position * 7) - 2) < 0.001:
            return 2.0
        elif abs((beat_position * 5) - 2) < 0.001:
            return 2.0
        else:
            return 0.0
    else:
        return 0.0


def pitch_into_vector(pitch: int) -> List[float]:
    """
    Convert MIDI pitch to 13-dimensional vector (octave + pitch class one-hot).

    Args:
        pitch: MIDI pitch value (0-127)

    Returns:
        13-dim vector: [normalized_octave, pitch_class_one_hot (12)]
    """
    pitch_vec = [0.0] * 13
    octave = (pitch // 12) - 1
    octave_normalized = (octave - 4) / 4  # Normalize around octave 4
    pitch_class = pitch % 12

    pitch_vec[0] = octave_normalized
    pitch_vec[pitch_class + 1] = 1.0

    return pitch_vec


def time_signature_to_vector(numerator: int, denominator: int) -> List[int]:
    """
    Convert time signature to 9-dimensional multi-hot vector.

    Args:
        numerator: Time signature numerator
        denominator: Time signature denominator

    Returns:
        9-dim vector: [numerator_encoding (5), denominator_encoding (4)]
    """
    denominator_list = [2, 4, 8, 16]
    numerator_vec = [0] * 5
    denominator_vec = [0] * 4

    # Denominator encoding
    if denominator == 32:
        denominator_vec[-1] = 1
    elif denominator in denominator_list:
        denominator_vec[denominator_list.index(denominator)] = 1

    # Numerator encoding (multi-hot for compound meters)
    if numerator == 2:
        numerator_vec[0] = 1
    elif numerator == 3:
        numerator_vec[1] = 1
    elif numerator == 4:
        numerator_vec[0] = 1
        numerator_vec[2] = 1
    elif numerator == 6:
        numerator_vec[0] = 1
        numerator_vec[3] = 1
    elif numerator == 9:
        numerator_vec[1] = 1
        numerator_vec[3] = 1
    elif numerator in [12, 24]:
        numerator_vec[0] = 1
        numerator_vec[2] = 1
        numerator_vec[3] = 1
    else:
        numerator_vec[4] = 1  # Unknown

    return numerator_vec + denominator_vec


def composer_name_to_vec(composer_name: str) -> List[int]:
    """
    Convert composer name to 17-dimensional one-hot vector.

    Args:
        composer_name: Composer name

    Returns:
        17-dim one-hot vector (16 composers + unknown)
    """
    composer_list = [
        "Bach",
        "Balakirev",
        "Beethoven",
        "Brahms",
        "Chopin",
        "Debussy",
        "Glinka",
        "Haydn",
        "Liszt",
        "Mozart",
        "Prokofiev",
        "Rachmaninoff",
        "Ravel",
        "Schubert",
        "Schumann",
        "Scriabin",
    ]
    one_hot = [0] * 17

    if composer_name in composer_list:
        one_hot[composer_list.index(composer_name)] = 1
    else:
        one_hot[-1] = 1  # Unknown

    return one_hot


def note_notation_to_vector(
    is_trill: bool = False,
    is_tenuto: bool = False,
    is_accent: bool = False,
    is_staccato: bool = False,
    is_fermata: bool = False,
    is_arpeggiate: bool = False,
    is_strong_accent: bool = False,
    is_cue: bool = False,
    is_slash: bool = False,
) -> List[int]:
    """
    Convert note notation flags to 9-dimensional multi-hot vector.

    Returns:
        9-dim multi-hot vector for notation marks
    """
    return [
        int(is_trill),
        int(is_tenuto),
        int(is_accent),
        int(is_staccato),
        int(is_fermata),
        int(is_arpeggiate),
        int(is_strong_accent),
        int(is_cue),
        int(is_slash),
    ]
