"""MIDI-to-graph conversion for GNN symbolic encoders (S2).

Converts MIDI files into PyTorch Geometric graph representations using a 4-edge-type
model (onset/during/follow/silence) based on GraphMuse research.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import torch
from torch_geometric.data import Data, HeteroData


EDGE_TYPE_ONSET = 0
EDGE_TYPE_DURING = 1
EDGE_TYPE_FOLLOW = 2
EDGE_TYPE_SILENCE = 3


def assign_voices(
    notes: list[pretty_midi.Note],
    max_voices: int = 8,
    max_gap: float = 2.0,
) -> list[int]:
    """Greedy voice assignment for polyphonic piano MIDI.

    Assigns each note to a voice based on temporal proximity and pitch distance.
    Notes are processed in onset order; each is assigned to the voice whose last
    note ended closest and had the smallest pitch distance.

    Args:
        notes: List of pretty_midi.Note objects, will be sorted by onset.
        max_voices: Maximum number of voices to allow.
        max_gap: Maximum gap (seconds) to consider reusing a voice.

    Returns:
        List of voice IDs (0-indexed) in the same order as input notes.
    """
    if not notes:
        return []

    sorted_indices = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch))
    voice_assignments = [0] * len(notes)

    # Each voice tracks (last_offset, last_pitch, voice_id)
    active_voices: list[tuple[float, int, int]] = []
    next_voice_id = 0

    for idx in sorted_indices:
        note = notes[idx]
        best_voice = -1
        best_score = float("inf")

        for i, (last_off, last_pitch, vid) in enumerate(active_voices):
            # Voice is still active (last note hasn't ended) -- skip
            if last_off > note.start + 1e-6:
                continue
            gap = note.start - last_off
            if gap > max_gap:
                continue
            # Score: weighted combination of time gap and pitch distance
            score = abs(gap) + abs(note.pitch - last_pitch) / 127.0
            if score < best_score:
                best_score = score
                best_voice = i

        if best_voice >= 0:
            vid = active_voices[best_voice][2]
            active_voices[best_voice] = (note.end, note.pitch, vid)
            voice_assignments[idx] = vid
        else:
            vid = next_voice_id
            if next_voice_id < max_voices - 1:
                next_voice_id += 1
            active_voices.append((note.end, note.pitch, vid))
            voice_assignments[idx] = vid

    return voice_assignments


def _build_edges(
    onsets: np.ndarray,
    offsets: np.ndarray,
    follow_tolerance: float = 0.05,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Build 4-type directed edges between notes.

    For each pair (u, v) where onset(u) <= onset(v):
      - onset:   on(u) == on(v)  (simultaneous, chord tones)
      - during:  on(u) < on(v) < off(u)  (overlapping)
      - follow:  |off(u) - on(v)| < tolerance  (succession)
      - silence: off(u) < on(v) and no note w with on(w) in [off(u), on(v)]

    Returns:
        edges: List of (src, dst) tuples.
        edge_types: List of edge type integers (0-3).
    """
    n = len(onsets)
    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []

    # Sort note indices by onset for efficient processing
    sorted_idx = np.argsort(onsets)
    sorted_onsets = onsets[sorted_idx]

    for i_pos in range(n):
        u = sorted_idx[i_pos]
        on_u = sorted_onsets[i_pos]
        off_u = offsets[u]

        for j_pos in range(i_pos + 1, n):
            v = sorted_idx[j_pos]
            on_v = sorted_onsets[j_pos]

            # Once onset(v) is far beyond offset(u), skip remaining
            if on_v > off_u + follow_tolerance:
                # Check for silence edge: off(u) < on(v) and no note between
                # Only connect to the very next note that starts after u ends
                if off_u < on_v:
                    # Check if there's any note w starting in [off(u), on(v))
                    gap_start = off_u
                    has_intervening = False
                    for k_pos in range(i_pos + 1, j_pos):
                        w = sorted_idx[k_pos]
                        if offsets[w] > gap_start and onsets[w] < on_v:
                            has_intervening = True
                            break
                    if not has_intervening:
                        edges.append((u, v))
                        edge_types.append(EDGE_TYPE_SILENCE)
                break

            if abs(on_u - on_v) < 1e-6:
                # onset edge: simultaneous
                edges.append((u, v))
                edge_types.append(EDGE_TYPE_ONSET)
            elif on_u < on_v < off_u:
                # during edge: overlap
                edges.append((u, v))
                edge_types.append(EDGE_TYPE_DURING)
            elif abs(off_u - on_v) < follow_tolerance:
                # follow edge: succession
                edges.append((u, v))
                edge_types.append(EDGE_TYPE_FOLLOW)
            elif off_u < on_v:
                # Potential silence edge
                gap_start = off_u
                has_intervening = False
                for k_pos in range(i_pos + 1, j_pos):
                    w = sorted_idx[k_pos]
                    if onsets[w] >= gap_start and onsets[w] < on_v:
                        has_intervening = True
                        break
                if not has_intervening:
                    edges.append((u, v))
                    edge_types.append(EDGE_TYPE_SILENCE)

    return edges, edge_types


def midi_to_graph(
    midi_path: str | Path,
    max_voices: int = 8,
    follow_tolerance: float = 0.05,
) -> Data:
    """Convert a MIDI file to a PyTorch Geometric Data object.

    Node features (6-dim): [pitch, velocity, onset, duration, pedal, voice]
    All features normalized to [0, 1].

    Edge types: onset (0), during (1), follow (2), silence (3).
    Edges are bidirectional.

    Args:
        midi_path: Path to MIDI file.
        max_voices: Maximum voices for voice assignment.
        follow_tolerance: Time tolerance (seconds) for follow edges.

    Returns:
        PyG Data object with x, edge_index, edge_type attributes.

    Raises:
        ValueError: If the MIDI file contains no notes.
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # Collect all notes across instruments
    all_notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)

    if not all_notes:
        raise ValueError(f"No notes found in {midi_path}")

    # Sort by onset then pitch
    all_notes.sort(key=lambda n: (n.start, n.pitch))

    total_duration = max(n.end for n in all_notes)
    if total_duration == 0:
        total_duration = 1.0  # avoid division by zero

    # Build CC64 (sustain pedal) lookup from all instruments
    pedal_events: list[tuple[float, bool]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for cc in instrument.control_changes:
            if cc.number == 64:
                pedal_events.append((cc.time, cc.value >= 64))
    pedal_events.sort(key=lambda x: x[0])

    def pedal_at(time: float) -> float:
        """Return 1.0 if sustain pedal is on at given time, else 0.0."""
        state = False
        for t, on in pedal_events:
            if t > time:
                break
            state = on
        return 1.0 if state else 0.0

    # Voice assignment
    voices = assign_voices(all_notes, max_voices=max_voices)
    max_voice_id = max(voices) if voices else 0
    voice_norm = max_voice_id if max_voice_id > 0 else 1

    # Node features: [pitch, velocity, onset, duration, pedal, voice]
    features = []
    onsets = np.zeros(len(all_notes))
    offsets = np.zeros(len(all_notes))

    for i, note in enumerate(all_notes):
        pitch = (note.pitch - 21) / 87.0  # Piano range A0(21) to C8(108)
        velocity = note.velocity / 127.0
        onset = note.start / total_duration
        duration = (note.end - note.start) / total_duration
        pedal = pedal_at(note.start)
        voice = voices[i] / voice_norm

        # Clamp to [0, 1]
        pitch = max(0.0, min(1.0, pitch))
        velocity = max(0.0, min(1.0, velocity))
        onset = max(0.0, min(1.0, onset))
        duration = max(0.0, min(1.0, duration))

        features.append([pitch, velocity, onset, duration, pedal, voice])
        onsets[i] = note.start
        offsets[i] = note.end

    x = torch.tensor(features, dtype=torch.float32)

    # Build edges
    edges, edge_types = _build_edges(onsets, offsets, follow_tolerance)

    # Make bidirectional
    bi_edges = []
    bi_types = []
    for (src, dst), etype in zip(edges, edge_types):
        bi_edges.append((src, dst))
        bi_edges.append((dst, src))
        bi_types.append(etype)
        bi_types.append(etype)

    if bi_edges:
        edge_index = torch.tensor(bi_edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(bi_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


def homo_to_hetero_graph(homo: Data) -> HeteroData:
    """Convert a homogeneous score graph to heterogeneous format.

    Splits edges by type into separate edge_index tensors for HeteroConv.
    Avoids re-parsing MIDI when you already have a homogeneous graph.

    Args:
        homo: PyG Data object from midi_to_graph() with edge_type attribute.

    Returns:
        PyG HeteroData object with per-type edge indices.
    """
    data = HeteroData()
    data["note"].x = homo.x

    type_names = ["onset", "during", "follow", "silence"]
    edge_index = homo.edge_index
    edge_type = homo.edge_type

    for tid, tname in enumerate(type_names):
        mask = edge_type == tid
        if mask.any():
            data["note", tname, "note"].edge_index = edge_index[:, mask]
        else:
            data["note", tname, "note"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )

    return data


def midi_to_hetero_graph(
    midi_path: str | Path,
    max_voices: int = 8,
    follow_tolerance: float = 0.05,
) -> HeteroData:
    """Convert a MIDI file to a heterogeneous PyG HeteroData object.

    Same node features as midi_to_graph(), but edges are stored per type
    as separate edge_index tensors for use with HeteroConv.

    Edge types stored as:
        ("note", "onset", "note"), ("note", "during", "note"),
        ("note", "follow", "note"), ("note", "silence", "note")

    Args:
        midi_path: Path to MIDI file.
        max_voices: Maximum voices for voice assignment.
        follow_tolerance: Time tolerance (seconds) for follow edges.

    Returns:
        PyG HeteroData object.

    Raises:
        ValueError: If the MIDI file contains no notes.
    """
    homo = midi_to_graph(midi_path, max_voices, follow_tolerance)
    return homo_to_hetero_graph(homo)


def sample_negative_edges(
    data: Data,
    num_neg: int,
) -> torch.Tensor:
    """Sample random negative edges (node pairs with no existing edge).

    Args:
        data: PyG Data object with edge_index.
        num_neg: Number of negative edges to sample.

    Returns:
        Tensor of shape [2, num_neg] with negative edge pairs.
    """
    num_nodes = data.x.size(0)
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    # Build set of existing edges for O(1) lookup
    edge_set: set[tuple[int, int]] = set()
    if data.edge_index.size(1) > 0:
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        for s, d in zip(src, dst):
            edge_set.add((s, d))

    neg_edges: list[tuple[int, int]] = []
    max_attempts = num_neg * 10
    attempts = 0

    while len(neg_edges) < num_neg and attempts < max_attempts:
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        attempts += 1
        if u == v:
            continue
        if (u, v) not in edge_set and (v, u) not in edge_set:
            neg_edges.append((u, v))
            edge_set.add((u, v))  # prevent duplicates

    if not neg_edges:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
