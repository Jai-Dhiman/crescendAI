"""ASAP dataset parsing utilities.

The ASAP dataset contains 1,067 performances of 236 classical piano pieces
with note-level alignments between scores and performances.

Reference: https://github.com/CPJKU/asap-dataset
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NoteAlignment:
    """A single note alignment between score and performance."""

    score_onset: float  # Onset time in score (beats or seconds depending on source)
    performance_onset: float  # Onset time in performance (seconds)
    pitch: int  # MIDI pitch number (0-127)
    velocity: int  # MIDI velocity (0-127)
    duration: float  # Note duration in the performance (seconds)
    score_duration: Optional[float] = None  # Duration in score


@dataclass
class ASAPPerformance:
    """Metadata for a single ASAP performance."""

    # Identifiers
    performance_id: str
    composer: str
    title: str
    performer: Optional[str] = None

    # File paths (relative to ASAP root)
    midi_score_path: Optional[Path] = None
    midi_performance_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    alignment_path: Optional[Path] = None  # Note alignments
    annotations_path: Optional[Path] = None  # Measure/beat annotations

    # Metadata
    year: Optional[int] = None
    source: Optional[str] = None  # e.g., "MAPS", "Vienna4x22"

    def has_audio(self) -> bool:
        """Check if this performance has audio."""
        return self.audio_path is not None

    def has_alignment(self) -> bool:
        """Check if this performance has note alignments."""
        return self.alignment_path is not None


@dataclass
class ASAPDatasetIndex:
    """Index for filtering and accessing ASAP performances."""

    performances: List[ASAPPerformance] = field(default_factory=list)
    _by_composer: Dict[str, List[ASAPPerformance]] = field(
        default_factory=dict, repr=False
    )
    _by_piece: Dict[str, List[ASAPPerformance]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self):
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices."""
        self._by_composer = {}
        self._by_piece = {}

        for perf in self.performances:
            # Index by composer
            if perf.composer not in self._by_composer:
                self._by_composer[perf.composer] = []
            self._by_composer[perf.composer].append(perf)

            # Index by piece (composer + title)
            piece_key = f"{perf.composer}/{perf.title}"
            if piece_key not in self._by_piece:
                self._by_piece[piece_key] = []
            self._by_piece[piece_key].append(perf)

    def add_performance(self, perf: ASAPPerformance):
        """Add a performance to the index."""
        self.performances.append(perf)
        self._build_indices()

    def filter_by_composer(self, composer: str) -> List[ASAPPerformance]:
        """Get all performances by a composer."""
        return self._by_composer.get(composer, [])

    def filter_by_piece(self, composer: str, title: str) -> List[ASAPPerformance]:
        """Get all performances of a specific piece."""
        piece_key = f"{composer}/{title}"
        return self._by_piece.get(piece_key, [])

    def get_multi_performer_pieces(
        self, min_performers: int = 2
    ) -> Dict[str, List[ASAPPerformance]]:
        """Get pieces with multiple performers (useful for disentanglement)."""
        return {
            piece: perfs
            for piece, perfs in self._by_piece.items()
            if len(perfs) >= min_performers
        }

    def get_composers(self) -> List[str]:
        """Get list of all composers."""
        return list(self._by_composer.keys())

    def get_pieces(self) -> List[str]:
        """Get list of all pieces (composer/title)."""
        return list(self._by_piece.keys())

    def filter_with_audio(self) -> List[ASAPPerformance]:
        """Get performances that have audio files."""
        return [p for p in self.performances if p.has_audio()]

    def filter_with_alignments(self) -> List[ASAPPerformance]:
        """Get performances that have note alignments."""
        return [p for p in self.performances if p.has_alignment()]

    def __len__(self) -> int:
        return len(self.performances)


def parse_asap_metadata(asap_root: Path) -> ASAPDatasetIndex:
    """Parse ASAP dataset metadata from the repository.

    Args:
        asap_root: Path to the cloned ASAP dataset repository.

    Returns:
        ASAPDatasetIndex containing all performances.

    Raises:
        FileNotFoundError: If asap_root or metadata files don't exist.
    """
    asap_root = Path(asap_root)
    if not asap_root.exists():
        raise FileNotFoundError(f"ASAP root not found: {asap_root}")

    # ASAP uses a metadata.json file
    metadata_path = asap_root / "asap_annotations.json"
    if not metadata_path.exists():
        # Try alternative location
        metadata_path = asap_root / "metadata.json"

    performances = []

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        # ASAP annotations is a dict with paths as keys
        if isinstance(metadata, dict):
            for path, annotations in metadata.items():
                perf = _parse_metadata_entry_from_path(path, annotations, asap_root)
                if perf:
                    performances.append(perf)
        else:
            # Legacy list format
            for entry in metadata:
                perf = _parse_metadata_entry(entry, asap_root)
                if perf:
                    performances.append(perf)
    else:
        # Fall back to directory structure parsing
        performances = _parse_from_directory_structure(asap_root)

    return ASAPDatasetIndex(performances=performances)


def _parse_metadata_entry_from_path(
    path: str, annotations: Dict, asap_root: Path
) -> Optional[ASAPPerformance]:
    """Parse a performance entry from ASAP path-keyed format.

    ASAP paths are structured as: Composer/Piece/.../PerformerXX.mid
    e.g., "Bach/Fugue/bwv_846/Shi05M.mid"
    """
    try:
        parts = path.split("/")
        if len(parts) < 2:
            return None

        composer = parts[0]
        # Title is everything between composer and the MIDI file
        title = "/".join(parts[1:-1]) if len(parts) > 2 else parts[1]
        midi_filename = parts[-1]

        # Extract performer from filename (remove .mid extension)
        performer = Path(midi_filename).stem

        # Look for corresponding score MIDI
        perf_dir = asap_root / Path(path).parent
        score_midi = None
        for score_name in ["midi_score.mid", "score.mid"]:
            score_path = perf_dir / score_name
            if score_path.exists():
                score_midi = Path(path).parent / score_name
                break

        # Look for alignment file in {performer}_note_alignments/note_alignment.tsv
        alignment_dir = perf_dir / f"{performer}_note_alignments"
        alignment_tsv = alignment_dir / "note_alignment.tsv"
        alignment_path = (
            Path(path).parent / f"{performer}_note_alignments" / "note_alignment.tsv"
            if alignment_tsv.exists()
            else None
        )

        # Look for annotations file ({performer}_annotations.txt)
        annotations_txt = perf_dir / f"{performer}_annotations.txt"
        annotations_path = (
            Path(path).parent / f"{performer}_annotations.txt"
            if annotations_txt.exists()
            else None
        )

        return ASAPPerformance(
            performance_id=path,
            composer=composer,
            title=title,
            performer=performer,
            midi_score_path=score_midi,
            midi_performance_path=Path(path),
            alignment_path=alignment_path,
            annotations_path=annotations_path,
        )
    except (KeyError, TypeError, IndexError):
        return None


def _parse_metadata_entry(entry: Dict, asap_root: Path) -> Optional[ASAPPerformance]:
    """Parse a single metadata entry from ASAP JSON."""
    try:
        # Extract paths relative to asap_root
        midi_score = entry.get("midi_score")
        midi_perf = entry.get("midi_performance") or entry.get("midi")
        audio = entry.get("audio")
        alignment = entry.get("alignment") or entry.get("note_alignment")
        annotations = entry.get("annotations") or entry.get("beat_annotations")

        return ASAPPerformance(
            performance_id=entry.get("id", entry.get("name", "")),
            composer=entry.get("composer", "Unknown"),
            title=entry.get("title", entry.get("work", "Unknown")),
            performer=entry.get("performer"),
            midi_score_path=Path(midi_score) if midi_score else None,
            midi_performance_path=Path(midi_perf) if midi_perf else None,
            audio_path=Path(audio) if audio else None,
            alignment_path=Path(alignment) if alignment else None,
            annotations_path=Path(annotations) if annotations else None,
            year=entry.get("year"),
            source=entry.get("source"),
        )
    except (KeyError, TypeError):
        return None


def _parse_from_directory_structure(asap_root: Path) -> List[ASAPPerformance]:
    """Parse ASAP dataset from directory structure when no metadata JSON exists."""
    performances = []

    # ASAP structure: composer/piece/performance_files
    for composer_dir in asap_root.iterdir():
        if not composer_dir.is_dir() or composer_dir.name.startswith("."):
            continue

        composer = composer_dir.name

        for piece_dir in composer_dir.iterdir():
            if not piece_dir.is_dir() or piece_dir.name.startswith("."):
                continue

            title = piece_dir.name

            # Find all MIDI performance files
            midi_files = list(piece_dir.glob("*.mid")) + list(piece_dir.glob("*.midi"))

            # Separate score from performances
            score_midi = None
            perf_midis = []

            for midi_file in midi_files:
                if "score" in midi_file.stem.lower():
                    score_midi = midi_file
                else:
                    perf_midis.append(midi_file)

            # Create performance entries
            for i, midi_perf in enumerate(perf_midis):
                perf_id = f"{composer}_{title}_{i}"

                # Look for corresponding files
                alignment_file = piece_dir / f"{midi_perf.stem}_alignment.txt"
                if not alignment_file.exists():
                    alignment_file = piece_dir / f"{midi_perf.stem}.txt"

                annotations_file = piece_dir / f"{midi_perf.stem}_annotations.txt"
                if not annotations_file.exists():
                    annotations_file = piece_dir / "annotations.txt"

                audio_file = piece_dir / f"{midi_perf.stem}.wav"
                if not audio_file.exists():
                    audio_file = piece_dir / f"{midi_perf.stem}.flac"

                perf = ASAPPerformance(
                    performance_id=perf_id,
                    composer=composer,
                    title=title,
                    midi_score_path=score_midi.relative_to(asap_root)
                    if score_midi
                    else None,
                    midi_performance_path=midi_perf.relative_to(asap_root),
                    audio_path=audio_file.relative_to(asap_root)
                    if audio_file.exists()
                    else None,
                    alignment_path=alignment_file.relative_to(asap_root)
                    if alignment_file.exists()
                    else None,
                    annotations_path=annotations_file.relative_to(asap_root)
                    if annotations_file.exists()
                    else None,
                )
                performances.append(perf)

    return performances


def load_note_alignments(
    alignment_path: Path,
    asap_root: Optional[Path] = None,
) -> List[NoteAlignment]:
    """Load note-level alignments from an alignment file.

    ASAP alignment files are typically TSV with columns:
    score_onset, performance_onset, pitch, velocity, duration

    Args:
        alignment_path: Path to the alignment file (relative or absolute).
        asap_root: If provided and alignment_path is relative, prepend this.

    Returns:
        List of NoteAlignment objects.

    Raises:
        FileNotFoundError: If alignment file doesn't exist.
    """
    if asap_root and not alignment_path.is_absolute():
        alignment_path = Path(asap_root) / alignment_path

    alignment_path = Path(alignment_path)
    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    alignments = []

    with open(alignment_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                parts = line.split()  # Try space-separated

            if len(parts) >= 5:
                try:
                    alignment = NoteAlignment(
                        score_onset=float(parts[0]),
                        performance_onset=float(parts[1]),
                        pitch=int(float(parts[2])),
                        velocity=int(float(parts[3])),
                        duration=float(parts[4]),
                        score_duration=float(parts[5]) if len(parts) > 5 else None,
                    )
                    alignments.append(alignment)
                except (ValueError, IndexError):
                    continue

    return alignments


def extract_onset_pairs(
    alignments: List[NoteAlignment],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract score and performance onset arrays from alignments.

    Args:
        alignments: List of NoteAlignment objects.

    Returns:
        Tuple of (score_onsets, performance_onsets) as numpy arrays.
    """
    if not alignments:
        return np.array([]), np.array([])

    score_onsets = np.array([a.score_onset for a in alignments])
    perf_onsets = np.array([a.performance_onset for a in alignments])

    return score_onsets, perf_onsets


def get_measure_boundaries(
    annotations_path: Path,
    asap_root: Optional[Path] = None,
) -> List[Tuple[float, float]]:
    """Load measure boundaries from annotation file.

    Args:
        annotations_path: Path to annotations file.
        asap_root: If provided and path is relative, prepend this.

    Returns:
        List of (start_time, end_time) tuples for each measure.

    Raises:
        FileNotFoundError: If annotations file doesn't exist.
    """
    if asap_root and not annotations_path.is_absolute():
        annotations_path = Path(asap_root) / annotations_path

    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    beat_times = []

    with open(annotations_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    time = float(parts[0])
                    # Parts[1] might be beat number or measure.beat notation
                    beat_times.append(time)
                except ValueError:
                    continue

    # Convert beat times to measure boundaries
    # Assume 4/4 time signature if not specified
    measures = []
    beats_per_measure = 4

    for i in range(0, len(beat_times) - beats_per_measure, beats_per_measure):
        start = beat_times[i]
        end = beat_times[min(i + beats_per_measure, len(beat_times) - 1)]
        measures.append((start, end))

    return measures


def get_performance_key(perf: ASAPPerformance) -> str:
    """Generate a unique key for a performance."""
    return perf.performance_id or f"{perf.composer}_{perf.title}"
