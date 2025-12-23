"""
Score alignment module for piano performance evaluation.

Implements score-to-performance alignment and feature extraction
following the PercePiano approach for comparing MIDI performances
to reference MusicXML scores.

Features extracted:
- Timing deviations (onset deviation from score)
- Tempo ratio (local tempo vs marked tempo)
- Dynamic deviations (velocity vs marked dynamics)
- Articulation deviations (duration ratio vs marked articulation)
"""

import numpy as np

# Number of features per note (expanded to match PercePiano-style features)
NUM_NOTE_FEATURES = 20
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import pretty_midi


@dataclass
class ScoreNote:
    """Represents a note from the musical score."""

    pitch: int  # MIDI pitch (21-108)
    onset_beat: float  # Onset time in beats
    duration_beat: float  # Duration in beats
    voice: int  # Voice/part number
    measure: int  # Measure number
    beat_in_measure: float  # Beat position within measure
    dynamic: Optional[str] = None  # Dynamic marking (pp, p, mp, mf, f, ff, etc.)
    articulation: Optional[str] = None  # Articulation (staccato, legato, etc.)
    tempo_marking: Optional[float] = None  # Tempo in BPM if marked


@dataclass
class AlignedNote:
    """A performance note aligned to its corresponding score note."""

    # Performance data
    perf_pitch: int
    perf_onset: float  # in seconds
    perf_duration: float  # in seconds
    perf_velocity: int

    # Score data (None if unmatched)
    score_pitch: Optional[int] = None
    score_onset_beat: Optional[float] = None
    score_duration_beat: Optional[float] = None
    score_voice: Optional[int] = None
    score_measure: Optional[int] = None
    score_dynamic: Optional[str] = None
    score_articulation: Optional[str] = None

    # Deviation features (computed)
    onset_deviation: Optional[float] = None  # in beats
    duration_ratio: Optional[float] = None  # perf_duration / expected_duration
    velocity_deviation: Optional[float] = None  # deviation from expected velocity

    # Extended features for HAN
    beat_index: Optional[int] = None  # Beat index within piece
    beat_position: Optional[float] = None  # Position within beat (0-1)
    local_tempo_ratio: Optional[float] = None  # Local tempo vs expected
    articulation_log: Optional[float] = (
        None  # Log-scale articulation (PercePiano style)
    )
    is_chord_member: bool = False  # Part of a chord
    following_rest: Optional[float] = None  # Duration of rest after note (in beats)


class MusicXMLParser:
    """
    Parser for MusicXML score files.

    Extracts note information including pitch, timing, dynamics,
    and articulation markings from MusicXML format scores.
    """

    # Dynamic marking to velocity mapping (approximate)
    DYNAMIC_TO_VELOCITY = {
        "ppp": 20,
        "pp": 35,
        "p": 50,
        "mp": 65,
        "mf": 80,
        "m": 75,
        "f": 95,
        "ff": 110,
        "fff": 125,
    }

    def __init__(self):
        self.divisions = 1  # Divisions per quarter note
        self.current_tempo = 120.0  # Default tempo

    def parse(self, musicxml_path: Union[str, Path]) -> List[ScoreNote]:
        """
        Parse a MusicXML file and extract note information.

        Args:
            musicxml_path: Path to MusicXML file

        Returns:
            List of ScoreNote objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ET.ParseError: If XML is malformed
        """
        path = Path(musicxml_path)
        if not path.exists():
            raise FileNotFoundError(f"MusicXML file not found: {path}")

        tree = ET.parse(path)
        root = tree.getroot()

        # Handle namespace if present
        ns = self._get_namespace(root)

        notes = []
        current_measure = 0
        current_beat = 0.0
        current_dynamic = None
        current_tempo = 120.0

        # Find all parts
        parts = root.findall(f".//{ns}part") if ns else root.findall(".//part")

        for part_idx, part in enumerate(parts):
            measures = part.findall(f"{ns}measure") if ns else part.findall("measure")

            for measure in measures:
                measure_num = int(measure.get("number", current_measure + 1))
                current_measure = measure_num
                measure_beat = 0.0

                # Get divisions (may be updated per measure)
                attributes = (
                    measure.find(f"{ns}attributes")
                    if ns
                    else measure.find("attributes")
                )
                if attributes is not None:
                    div_elem = (
                        attributes.find(f"{ns}divisions")
                        if ns
                        else attributes.find("divisions")
                    )
                    if div_elem is not None:
                        self.divisions = int(div_elem.text)

                # Process elements in order
                for elem in measure:
                    tag = elem.tag.replace(f"{{{ns}}}", "") if ns else elem.tag

                    if tag == "direction":
                        # Check for tempo and dynamics
                        tempo, dynamic = self._parse_direction(elem, ns)
                        if tempo is not None:
                            current_tempo = tempo
                        if dynamic is not None:
                            current_dynamic = dynamic

                    elif tag == "note":
                        note_info = self._parse_note(
                            elem,
                            ns,
                            current_beat + measure_beat,
                            current_measure,
                            measure_beat,
                            part_idx,
                            current_dynamic,
                            current_tempo,
                        )

                        if note_info is not None:
                            notes.append(note_info)

                        # Update beat position
                        duration = self._get_note_duration(elem, ns)
                        if not self._is_chord(elem, ns):
                            measure_beat += duration

                # Update global beat counter
                current_beat += measure_beat

        return notes

    def _get_namespace(self, root) -> str:
        """Extract namespace from root element if present."""
        if root.tag.startswith("{"):
            return root.tag[1 : root.tag.index("}")]
        return ""

    def _parse_note(
        self,
        note_elem,
        ns: str,
        current_beat: float,
        measure: int,
        beat_in_measure: float,
        voice: int,
        current_dynamic: Optional[str],
        current_tempo: float,
    ) -> Optional[ScoreNote]:
        """Parse a single note element."""
        # Skip rests
        rest = note_elem.find(f"{ns}rest") if ns else note_elem.find("rest")
        if rest is not None:
            return None

        # Get pitch
        pitch_elem = note_elem.find(f"{ns}pitch") if ns else note_elem.find("pitch")
        if pitch_elem is None:
            return None

        step = pitch_elem.find(f"{ns}step") if ns else pitch_elem.find("step")
        octave = pitch_elem.find(f"{ns}octave") if ns else pitch_elem.find("octave")
        alter = pitch_elem.find(f"{ns}alter") if ns else pitch_elem.find("alter")

        if step is None or octave is None:
            return None

        midi_pitch = self._note_to_midi(
            step.text, int(octave.text), int(alter.text) if alter is not None else 0
        )

        # Get duration in beats
        duration_beats = self._get_note_duration(note_elem, ns)

        # Get articulation
        articulation = self._get_articulation(note_elem, ns)

        # Get voice (for multi-voice parts)
        voice_elem = note_elem.find(f"{ns}voice") if ns else note_elem.find("voice")
        note_voice = int(voice_elem.text) if voice_elem is not None else voice

        return ScoreNote(
            pitch=midi_pitch,
            onset_beat=current_beat,
            duration_beat=duration_beats,
            voice=note_voice,
            measure=measure,
            beat_in_measure=beat_in_measure,
            dynamic=current_dynamic,
            articulation=articulation,
            tempo_marking=current_tempo,
        )

    def _get_note_duration(self, note_elem, ns: str) -> float:
        """
        Get note duration in beats.

        Grace notes in MusicXML don't have duration elements - this is valid.
        Returns a small default duration (0.125 beats) for grace notes.
        """
        duration_elem = (
            note_elem.find(f"{ns}duration") if ns else note_elem.find("duration")
        )
        if duration_elem is None:
            # Check if this is a grace note (grace notes don't have duration in MusicXML)
            grace_elem = note_elem.find(f"{ns}grace") if ns else note_elem.find("grace")
            if grace_elem is not None:
                # Grace note - return small default duration
                return 0.125  # 1/8 beat for grace notes
            # Not a grace note but missing duration - use default with warning
            # This can happen with some MusicXML exporters
            return 0.25  # Default to quarter beat
        return int(duration_elem.text) / self.divisions

    def _is_chord(self, note_elem, ns: str) -> bool:
        """Check if note is part of a chord (doesn't advance time)."""
        chord = note_elem.find(f"{ns}chord") if ns else note_elem.find("chord")
        return chord is not None

    def _note_to_midi(self, step: str, octave: int, alter: int = 0) -> int:
        """Convert note name to MIDI pitch."""
        note_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        return note_map[step] + alter + (octave + 1) * 12

    def _parse_direction(
        self, direction_elem, ns: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Parse direction element for tempo and dynamics."""
        tempo = None
        dynamic = None

        # Look for tempo
        sound = (
            direction_elem.find(f".//{ns}sound")
            if ns
            else direction_elem.find(".//sound")
        )
        if sound is not None and "tempo" in sound.attrib:
            tempo = float(sound.attrib["tempo"])

        # Look for dynamics
        dynamics_elem = (
            direction_elem.find(f".//{ns}dynamics")
            if ns
            else direction_elem.find(".//dynamics")
        )
        if dynamics_elem is not None:
            for dyn in dynamics_elem:
                tag = dyn.tag.replace(f"{{{ns}}}", "") if ns else dyn.tag
                if tag in self.DYNAMIC_TO_VELOCITY:
                    dynamic = tag
                    break

        return tempo, dynamic

    def _get_articulation(self, note_elem, ns: str) -> Optional[str]:
        """Extract articulation from note notations."""
        notations = (
            note_elem.find(f"{ns}notations") if ns else note_elem.find("notations")
        )
        if notations is None:
            return None

        articulations = (
            notations.find(f"{ns}articulations")
            if ns
            else notations.find("articulations")
        )
        if articulations is None:
            return None

        # Check for common articulations
        for art_type in ["staccato", "staccatissimo", "tenuto", "accent", "marcato"]:
            art = (
                articulations.find(f"{ns}{art_type}")
                if ns
                else articulations.find(art_type)
            )
            if art is not None:
                return art_type

        return None


class ScorePerformanceAligner:
    """
    Aligns performance MIDI notes to score notes using dynamic time warping.

    Based on the alignment approach used in PercePiano/VirtuosoNet.
    """

    def __init__(
        self,
        pitch_tolerance: int = 0,
        max_time_deviation: float = 2.0,  # Maximum deviation in beats
    ):
        """
        Args:
            pitch_tolerance: Allow matching notes with pitch difference <= tolerance
            max_time_deviation: Maximum allowed time deviation for matching (in beats)
        """
        self.pitch_tolerance = pitch_tolerance
        self.max_time_deviation = max_time_deviation

    def align(
        self,
        performance_midi: pretty_midi.PrettyMIDI,
        score_notes: List[ScoreNote],
        tempo_bpm: float = 120.0,
    ) -> List[AlignedNote]:
        """
        Align performance MIDI to score notes.

        Uses a greedy matching algorithm that pairs each performance note
        with the closest unmatched score note of the same pitch within
        a time window.

        Args:
            performance_midi: Performance MIDI data
            score_notes: List of ScoreNote from the score
            tempo_bpm: Tempo to use for beat-to-second conversion

        Returns:
            List of AlignedNote with computed deviations
        """
        # Extract performance notes
        perf_notes = []
        for instrument in performance_midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                perf_notes.append(
                    {
                        "pitch": note.pitch,
                        "onset": note.start,
                        "duration": note.end - note.start,
                        "velocity": note.velocity,
                    }
                )

        # Sort by onset time
        perf_notes.sort(key=lambda x: x["onset"])
        score_notes_list = sorted(score_notes, key=lambda x: x.onset_beat)

        # Convert tempo to seconds per beat
        sec_per_beat = 60.0 / tempo_bpm

        # Track which score notes have been matched
        matched_score_indices = set()

        aligned_notes = []

        for perf in perf_notes:
            # Convert performance onset to beats
            perf_onset_beat = perf["onset"] / sec_per_beat

            # Find best matching score note
            best_match_idx = None
            best_match_dist = float("inf")

            for i, score_note in enumerate(score_notes_list):
                if i in matched_score_indices:
                    continue

                # Check pitch match
                if abs(score_note.pitch - perf["pitch"]) > self.pitch_tolerance:
                    continue

                # Check time proximity
                time_diff = abs(perf_onset_beat - score_note.onset_beat)
                if time_diff > self.max_time_deviation:
                    continue

                if time_diff < best_match_dist:
                    best_match_dist = time_diff
                    best_match_idx = i

            # Create aligned note
            if best_match_idx is not None:
                matched_score_indices.add(best_match_idx)
                score_note = score_notes_list[best_match_idx]

                # Compute deviation features
                onset_deviation = perf_onset_beat - score_note.onset_beat

                # Expected duration in seconds
                expected_duration = score_note.duration_beat * sec_per_beat
                duration_ratio = (
                    perf["duration"] / expected_duration
                    if expected_duration > 0
                    else 1.0
                )

                # Expected velocity from dynamics
                expected_velocity = self._dynamic_to_velocity(score_note.dynamic)
                velocity_deviation = (
                    perf["velocity"] - expected_velocity if expected_velocity else 0.0
                )

                aligned_notes.append(
                    AlignedNote(
                        perf_pitch=perf["pitch"],
                        perf_onset=perf["onset"],
                        perf_duration=perf["duration"],
                        perf_velocity=perf["velocity"],
                        score_pitch=score_note.pitch,
                        score_onset_beat=score_note.onset_beat,
                        score_duration_beat=score_note.duration_beat,
                        score_voice=score_note.voice,
                        score_measure=score_note.measure,
                        score_dynamic=score_note.dynamic,
                        score_articulation=score_note.articulation,
                        onset_deviation=onset_deviation,
                        duration_ratio=duration_ratio,
                        velocity_deviation=velocity_deviation,
                    )
                )
            else:
                # Unmatched performance note (extra note or wrong note)
                aligned_notes.append(
                    AlignedNote(
                        perf_pitch=perf["pitch"],
                        perf_onset=perf["onset"],
                        perf_duration=perf["duration"],
                        perf_velocity=perf["velocity"],
                    )
                )

        return aligned_notes

    def _dynamic_to_velocity(self, dynamic: Optional[str]) -> Optional[int]:
        """Convert dynamic marking to expected velocity."""
        if dynamic is None:
            return None
        return MusicXMLParser.DYNAMIC_TO_VELOCITY.get(dynamic)


class ScoreAlignmentFeatureExtractor:
    """
    Extracts score alignment features for piano performance evaluation.

    Expanded feature set (30+ features per note) following PercePiano approach:

    Note-level features:
    - Timing: onset_deviation, beat_position, local_tempo_ratio
    - Articulation: duration_ratio, articulation_log
    - Dynamics: velocity, velocity_deviation
    - Pitch: midi_pitch, pitch_class, octave
    - Context: voice, measure, beat_index, is_chord_member
    - Performance: matched indicator

    Also extracts note_locations for hierarchical processing:
    - beat: Beat index per note
    - measure: Measure index per note
    - voice: Voice index per note

    Global features (12):
    - Mean/std of deviations
    - Match rate
    - Quantiles
    - Reference tempo
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Number of notes for local tempo estimation
        """
        self.window_size = window_size
        self.parser = MusicXMLParser()
        self.aligner = ScorePerformanceAligner()

    def extract_features(
        self,
        performance_midi: pretty_midi.PrettyMIDI,
        score_path: Union[str, Path],
        tempo_bpm: float = 120.0,
    ) -> Dict[str, np.ndarray]:
        """
        Extract score alignment features from performance and score.

        Args:
            performance_midi: Performance MIDI data
            score_path: Path to MusicXML score file
            tempo_bpm: Reference tempo in BPM

        Returns:
            Dictionary of feature arrays:
            - note_features: [num_notes, 20] per-note features (expanded)
            - global_features: [12] aggregated statistics
            - tempo_curve: [num_segments] local tempo ratios
            - note_locations: Dict with beat/measure/voice indices
        """
        # Parse score
        score_notes = self.parser.parse(score_path)

        # Align performance to score
        aligned_notes = self.aligner.align(performance_midi, score_notes, tempo_bpm)

        # Enhance aligned notes with additional features
        aligned_notes = self._enhance_aligned_notes(aligned_notes, tempo_bpm)

        # Extract per-note features (expanded)
        note_features = self._extract_note_features(aligned_notes)

        # Compute global statistics
        global_features = self._extract_global_features(aligned_notes, tempo_bpm)

        # Compute tempo curve
        tempo_curve = self._extract_tempo_curve(aligned_notes, tempo_bpm)

        # Extract note_locations for hierarchical processing
        note_locations = self._extract_note_locations(aligned_notes)

        return {
            "note_features": note_features,
            "global_features": global_features,
            "tempo_curve": tempo_curve,
            "note_locations": note_locations,
        }

    def _enhance_aligned_notes(
        self,
        aligned_notes: List[AlignedNote],
        tempo_bpm: float,
    ) -> List[AlignedNote]:
        """
        Add extended features to aligned notes.

        Computes:
        - beat_index: Which beat the note falls on
        - beat_position: Position within the beat (0-1)
        - local_tempo_ratio: Local tempo vs expected
        - articulation_log: Log-scale articulation ratio
        - is_chord_member: Whether note is part of a chord
        - following_rest: Duration of rest after note
        """
        sec_per_beat = 60.0 / tempo_bpm

        # Sort by onset for chord detection and tempo estimation
        sorted_notes = sorted(
            [
                (i, n)
                for i, n in enumerate(aligned_notes)
                if n.score_onset_beat is not None
            ],
            key=lambda x: x[1].score_onset_beat,
        )

        # Detect chords (notes starting at same beat)
        chord_groups = {}
        for i, note in sorted_notes:
            beat = round(note.score_onset_beat * 4) / 4  # Quantize to 16th notes
            if beat not in chord_groups:
                chord_groups[beat] = []
            chord_groups[beat].append(i)

        # Mark chord members
        for indices in chord_groups.values():
            if len(indices) > 1:
                for idx in indices:
                    aligned_notes[idx].is_chord_member = True

        # Compute local tempo ratios in windows
        matched_indices = [
            i for i, n in enumerate(aligned_notes) if n.onset_deviation is not None
        ]
        for i, idx in enumerate(matched_indices):
            note = aligned_notes[idx]

            # Beat index and position
            if note.score_onset_beat is not None:
                note.beat_index = int(note.score_onset_beat)
                note.beat_position = note.score_onset_beat - note.beat_index

            # Log-scale articulation (PercePiano style)
            if note.duration_ratio is not None and note.duration_ratio > 0:
                note.articulation_log = float(np.log10(max(note.duration_ratio, 0.01)))
            else:
                note.articulation_log = 0.0

            # Local tempo ratio from window
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(len(matched_indices), i + self.window_size // 2)

            if end_idx > start_idx + 1:
                window_indices = matched_indices[start_idx:end_idx]
                first_note = aligned_notes[window_indices[0]]
                last_note = aligned_notes[window_indices[-1]]

                if (
                    first_note.perf_onset is not None
                    and last_note.perf_onset is not None
                    and first_note.score_onset_beat is not None
                    and last_note.score_onset_beat is not None
                ):
                    perf_ioi = last_note.perf_onset - first_note.perf_onset
                    score_ioi_beats = (
                        last_note.score_onset_beat - first_note.score_onset_beat
                    )

                    if score_ioi_beats > 0 and perf_ioi > 0:
                        expected_ioi = score_ioi_beats * sec_per_beat
                        note.local_tempo_ratio = expected_ioi / perf_ioi
                    else:
                        note.local_tempo_ratio = 1.0
                else:
                    note.local_tempo_ratio = 1.0
            else:
                note.local_tempo_ratio = 1.0

        # Compute following rest duration
        for i in range(len(sorted_notes) - 1):
            curr_idx, curr_note = sorted_notes[i]
            next_idx, next_note = sorted_notes[i + 1]

            if (
                curr_note.score_onset_beat is not None
                and curr_note.score_duration_beat is not None
            ):
                curr_end = curr_note.score_onset_beat + curr_note.score_duration_beat
                rest_duration = next_note.score_onset_beat - curr_end
                aligned_notes[curr_idx].following_rest = max(0.0, rest_duration)

        return aligned_notes

    def _extract_note_features(self, aligned_notes: List[AlignedNote]) -> np.ndarray:
        """
        Extract expanded per-note deviation features (20 features per note).

        Features:
        0: onset_deviation - timing deviation in beats
        1: duration_ratio - performed/expected duration
        2: articulation_log - log10(duration_ratio) for PercePiano style
        3: velocity - MIDI velocity (0-127 normalized to 0-1)
        4: velocity_deviation - deviation from expected velocity (normalized)
        5: local_tempo_ratio - local tempo vs marked tempo
        6: midi_pitch - MIDI pitch (normalized 0-1, 21-108 range)
        7: pitch_class - pitch class (0-11, normalized)
        8: octave - octave number (normalized)
        9: beat_position - position within beat (0-1)
        10: beat_index - beat index (normalized by max)
        11: measure - measure number (normalized by max)
        12: voice - voice number (normalized)
        13: matched - whether note was matched to score
        14: is_chord_member - whether note is part of chord
        15: following_rest - duration of rest after note (normalized)
        16: is_staccato - articulation indicator
        17: is_legato - articulation indicator
        18: dynamic_level - expected dynamic level (0-1)
        19: tempo_marking - tempo marking (log-normalized)
        """
        features = []

        # Get max values for normalization
        max_beat = max((n.beat_index or 0 for n in aligned_notes), default=1) or 1
        max_measure = max((n.score_measure or 0 for n in aligned_notes), default=1) or 1
        max_voice = max((n.score_voice or 1 for n in aligned_notes), default=1) or 1

        for note in aligned_notes:
            # Basic deviations
            onset_dev = (
                note.onset_deviation if note.onset_deviation is not None else 0.0
            )
            dur_ratio = note.duration_ratio if note.duration_ratio is not None else 1.0
            art_log = (
                note.articulation_log if note.articulation_log is not None else 0.0
            )
            vel = note.perf_velocity / 127.0
            vel_dev = (
                note.velocity_deviation or 0.0
            ) / 64.0  # Normalize to roughly -1 to 1
            local_tempo = (
                note.local_tempo_ratio if note.local_tempo_ratio is not None else 1.0
            )

            # Pitch features
            midi_pitch = (note.perf_pitch - 21) / 87.0  # Normalize 21-108 to 0-1
            pitch_class = (note.perf_pitch % 12) / 11.0
            octave = (note.perf_pitch // 12 - 1) / 8.0  # Normalize roughly 0-8 octaves

            # Position features
            beat_pos = note.beat_position if note.beat_position is not None else 0.0
            beat_idx = (note.beat_index or 0) / max_beat
            measure = (note.score_measure or 0) / max_measure
            voice = ((note.score_voice or 1) - 1) / max(max_voice - 1, 1)

            # Indicators
            matched = 1.0 if note.score_pitch is not None else 0.0
            is_chord = 1.0 if note.is_chord_member else 0.0
            following_rest = (
                min(note.following_rest or 0.0, 4.0) / 4.0
            )  # Clip and normalize

            # Articulation indicators
            is_staccato = (
                1.0 if note.score_articulation in ["staccato", "staccatissimo"] else 0.0
            )
            is_legato = 1.0 if note.score_articulation in ["tenuto", "legato"] else 0.0

            # Dynamic level from marking
            dynamic_map = {
                "ppp": 0.1,
                "pp": 0.2,
                "p": 0.35,
                "mp": 0.5,
                "mf": 0.65,
                "f": 0.8,
                "ff": 0.9,
                "fff": 1.0,
            }
            dynamic_level = dynamic_map.get(note.score_dynamic, 0.5)

            # Tempo marking (log-normalized)
            tempo_val = 0.5  # default
            # tempo_marking would come from score

            note_feat = [
                onset_dev,  # 0
                dur_ratio,  # 1
                art_log,  # 2
                vel,  # 3
                vel_dev,  # 4
                local_tempo,  # 5
                midi_pitch,  # 6
                pitch_class,  # 7
                octave,  # 8
                beat_pos,  # 9
                beat_idx,  # 10
                measure,  # 11
                voice,  # 12
                matched,  # 13
                is_chord,  # 14
                following_rest,  # 15
                is_staccato,  # 16
                is_legato,  # 17
                dynamic_level,  # 18
                tempo_val,  # 19
            ]
            features.append(note_feat)

        if not features:
            return np.zeros((1, NUM_NOTE_FEATURES), dtype=np.float32)

        return np.array(features, dtype=np.float32)

    def _extract_note_locations(
        self, aligned_notes: List[AlignedNote]
    ) -> Dict[str, np.ndarray]:
        """
        Extract note_locations for hierarchical processing (HAN encoder).

        Returns:
            Dict with:
            - beat: [num_notes] beat index per note (for note->beat aggregation)
            - measure: [num_notes] measure index per note (for beat->measure aggregation)
            - voice: [num_notes] voice index per note (for voice processing)

        Note: All indices are 1-based (0 is reserved for padding in HAN encoder).
        """
        beats = []
        measures = []
        voices = []

        for note in aligned_notes:
            # Beat index (1-based, default to 1 if not matched)
            # HAN encoder uses 0 for padding, so valid indices start from 1
            beat_idx = (note.beat_index + 1) if note.beat_index is not None else 1
            beats.append(beat_idx)

            # Measure index (1-based)
            measure_idx = (
                (note.score_measure + 1) if note.score_measure is not None else 1
            )
            measures.append(measure_idx)

            # Voice index (already 1-indexed for PercePiano compatibility)
            voice_idx = note.score_voice if note.score_voice is not None else 1
            voices.append(voice_idx)

        return {
            "beat": np.array(beats, dtype=np.int64),
            "measure": np.array(measures, dtype=np.int64),
            "voice": np.array(voices, dtype=np.int64),
        }

    def _extract_global_features(
        self,
        aligned_notes: List[AlignedNote],
        tempo_bpm: float,
    ) -> np.ndarray:
        """Extract aggregated global features."""
        # Filter matched notes
        matched = [n for n in aligned_notes if n.onset_deviation is not None]

        if not matched:
            return np.zeros(12, dtype=np.float32)

        onset_devs = [n.onset_deviation for n in matched]
        dur_ratios = [n.duration_ratio for n in matched]
        vel_devs = [
            n.velocity_deviation for n in matched if n.velocity_deviation is not None
        ]

        # Compute statistics
        features = [
            np.mean(onset_devs),
            np.std(onset_devs),
            np.mean(dur_ratios),
            np.std(dur_ratios),
            np.mean(vel_devs) if vel_devs else 0.0,
            np.std(vel_devs) if vel_devs else 0.0,
            len(matched) / len(aligned_notes),  # Match rate
            np.percentile(onset_devs, 25),  # Q1 onset deviation
            np.percentile(onset_devs, 75),  # Q3 onset deviation
            np.max(np.abs(onset_devs)),  # Max absolute onset deviation
            np.median(dur_ratios),  # Median duration ratio
            tempo_bpm,  # Reference tempo
        ]

        return np.array(features, dtype=np.float32)

    def _extract_tempo_curve(
        self,
        aligned_notes: List[AlignedNote],
        tempo_bpm: float,
    ) -> np.ndarray:
        """
        Extract local tempo ratio curve.

        Computes tempo ratio in sliding windows over the performance.
        """
        matched = [
            n
            for n in aligned_notes
            if n.onset_deviation is not None and n.score_onset_beat is not None
        ]

        if len(matched) < self.window_size:
            return np.ones(1, dtype=np.float32)

        # Sort by performance onset
        matched.sort(key=lambda x: x.perf_onset)

        tempo_ratios = []
        sec_per_beat = 60.0 / tempo_bpm

        for i in range(len(matched) - self.window_size):
            window = matched[i : i + self.window_size]

            # Compute local tempo from IOI (inter-onset interval)
            perf_ioi = window[-1].perf_onset - window[0].perf_onset
            score_ioi_beats = window[-1].score_onset_beat - window[0].score_onset_beat

            if score_ioi_beats > 0 and perf_ioi > 0:
                expected_ioi = score_ioi_beats * sec_per_beat
                local_tempo_ratio = expected_ioi / perf_ioi
                tempo_ratios.append(local_tempo_ratio)
            else:
                tempo_ratios.append(1.0)

        if not tempo_ratios:
            return np.ones(1, dtype=np.float32)

        return np.array(tempo_ratios, dtype=np.float32)


def load_score_midi(score_path: Union[str, Path]) -> pretty_midi.PrettyMIDI:
    """
    Load a 'score MIDI' file - a MIDI file that represents the exact
    notation from the score without expressive timing/dynamics.

    This is an alternative to parsing MusicXML directly.

    Args:
        score_path: Path to score MIDI file

    Returns:
        PrettyMIDI object representing the score
    """
    return pretty_midi.PrettyMIDI(str(score_path))


if __name__ == "__main__":
    print("Score alignment module loaded successfully")
    print("Features:")
    print("- MusicXML parsing with dynamics and articulation")
    print("- Score-to-performance note alignment")
    print("- Timing, duration, and velocity deviation extraction")
    print("- Local tempo curve estimation")
