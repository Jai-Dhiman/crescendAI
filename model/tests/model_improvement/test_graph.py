"""Tests for MIDI-to-graph conversion pipeline and graph dataset classes."""

import tempfile
from pathlib import Path

import pretty_midi
import pytest
import torch

from model_improvement.graph import (
    EDGE_TYPE_DURING,
    EDGE_TYPE_FOLLOW,
    EDGE_TYPE_ONSET,
    EDGE_TYPE_SILENCE,
    assign_voices,
    midi_to_graph,
    midi_to_hetero_graph,
    sample_negative_edges,
)
from model_improvement.data import (
    ScoreGraphPretrainingDataset,
    HeteroPretrainDataset,
    graph_pair_collate_fn,
    hetero_graph_collate_fn,
)


def _make_midi(notes: list[tuple[int, int, float, float]], tempo: float = 120.0) -> Path:
    """Create a temporary MIDI file from note specs.

    Args:
        notes: List of (pitch, velocity, start, end) tuples.
        tempo: BPM for the MIDI file.

    Returns:
        Path to the temporary MIDI file.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    for pitch, velocity, start, end in notes:
        piano.notes.append(pretty_midi.Note(
            velocity=velocity, pitch=pitch, start=start, end=end
        ))
    midi.instruments.append(piano)
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    midi.write(tmp.name)
    return Path(tmp.name)


class TestAssignVoices:
    def test_single_note(self):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        voices = assign_voices(piano.notes)
        assert voices == [0]

    def test_two_voices_alternating(self):
        """Two non-overlapping streams at different pitches."""
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        # Voice 0: low notes
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=40, start=0.0, end=0.5))
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=42, start=0.5, end=1.0))
        # Voice 1: high notes (simultaneous with voice 0)
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=80, start=0.0, end=0.5))
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=82, start=0.5, end=1.0))

        voices = assign_voices(piano.notes)
        # Notes at same time should get different voices
        assert len(set(voices)) == 2

    def test_empty_notes(self):
        assert assign_voices([]) == []

    def test_max_voices_cap(self):
        """Voices should not exceed max_voices."""
        notes = []
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        # Create 12 simultaneous notes
        for i in range(12):
            piano.notes.append(
                pretty_midi.Note(velocity=80, pitch=40 + i * 5, start=0.0, end=1.0)
            )
        voices = assign_voices(piano.notes, max_voices=4)
        assert max(voices) <= 3  # 0-indexed, max_voices=4


class TestMidiToGraph:
    def test_chord_graph(self):
        """C-E-G chord: 3 simultaneous notes should produce onset edges."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),  # C4
            (64, 80, 0.0, 1.0),  # E4
            (67, 80, 0.0, 1.0),  # G4
        ])
        data = midi_to_graph(path)

        # 3 nodes
        assert data.x.shape[0] == 3
        assert data.x.shape[1] == 6

        # All features should be in [0, 1]
        assert data.x.min() >= 0.0
        assert data.x.max() <= 1.0

        # Should have onset edges (simultaneous notes)
        assert data.edge_type is not None
        onset_mask = data.edge_type == EDGE_TYPE_ONSET
        assert onset_mask.any(), "Simultaneous notes should produce onset edges"

        # Edges should be bidirectional
        assert data.edge_index.shape[1] % 2 == 0

    def test_sequential_notes_follow_edges(self):
        """Three sequential notes should produce follow edges."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 1.0, 2.0),
            (67, 80, 2.0, 3.0),
        ])
        data = midi_to_graph(path)
        assert data.x.shape[0] == 3

        follow_mask = data.edge_type == EDGE_TYPE_FOLLOW
        assert follow_mask.any(), "Sequential notes should produce follow edges"

    def test_overlapping_notes_during_edges(self):
        """Overlapping notes should produce during edges."""
        path = _make_midi([
            (60, 80, 0.0, 2.0),  # Long note
            (64, 80, 0.5, 1.5),  # Starts during first note
        ])
        data = midi_to_graph(path)
        assert data.x.shape[0] == 2

        during_mask = data.edge_type == EDGE_TYPE_DURING
        assert during_mask.any(), "Overlapping notes should produce during edges"

    def test_silence_edges(self):
        """Notes with a gap should produce silence edges."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 2.0, 3.0),  # 1s gap
        ])
        data = midi_to_graph(path)
        assert data.x.shape[0] == 2

        silence_mask = data.edge_type == EDGE_TYPE_SILENCE
        assert silence_mask.any(), "Notes with gap should produce silence edges"

    def test_bidirectional_edges(self):
        """Every edge (u,v) should have a reverse (v,u)."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 1.0, 2.0),
            (67, 80, 2.0, 3.0),
        ])
        data = midi_to_graph(path)

        edge_set = set()
        for i in range(data.edge_index.shape[1]):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            edge_set.add((src, dst))

        for src, dst in list(edge_set):
            assert (dst, src) in edge_set, f"Edge ({src},{dst}) has no reverse"

    def test_node_features_normalized(self):
        """All node features should be in [0, 1]."""
        path = _make_midi([
            (21, 1, 0.0, 0.5),    # Lowest piano note, min velocity
            (108, 127, 0.5, 1.0),  # Highest piano note, max velocity
        ])
        data = midi_to_graph(path)
        assert data.x.min() >= 0.0
        assert data.x.max() <= 1.0

    def test_pedal_feature(self):
        """Pedal feature should reflect CC64 state."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=1.0, end=2.0))
        # Sustain pedal on at 0.5s
        piano.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=0.5))
        midi.instruments.append(piano)

        tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        midi.write(tmp.name)

        data = midi_to_graph(tmp.name)
        # Second note (at t=1.0) should have pedal=1.0 since CC64 was set at 0.5
        assert data.x[1, 4].item() == 1.0

    def test_empty_midi_raises(self):
        """MIDI with no notes should raise ValueError."""
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        midi.instruments.append(piano)
        tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        midi.write(tmp.name)

        with pytest.raises(ValueError, match="No notes found"):
            midi_to_graph(tmp.name)


class TestSampleNegativeEdges:
    def test_returns_correct_shape(self):
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),
            (67, 80, 0.0, 1.0),
            (72, 80, 1.0, 2.0),
            (76, 80, 1.0, 2.0),
        ])
        data = midi_to_graph(path)
        neg = sample_negative_edges(data, num_neg=5)
        assert neg.shape[0] == 2
        assert neg.shape[1] <= 5

    def test_no_overlap_with_existing_edges(self):
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),
            (67, 80, 0.0, 1.0),
            (72, 80, 1.0, 2.0),
        ])
        data = midi_to_graph(path)
        neg = sample_negative_edges(data, num_neg=3)

        existing = set()
        for i in range(data.edge_index.shape[1]):
            s = data.edge_index[0, i].item()
            d = data.edge_index[1, i].item()
            existing.add((s, d))

        for i in range(neg.shape[1]):
            s = neg[0, i].item()
            d = neg[1, i].item()
            assert (s, d) not in existing

    def test_single_node_returns_empty(self):
        """Graph with 1 node can't have negative edges."""
        path = _make_midi([(60, 80, 0.0, 1.0)])
        data = midi_to_graph(path)
        neg = sample_negative_edges(data, num_neg=5)
        assert neg.shape[1] == 0


class TestMidiToHeteroGraph:
    def test_hetero_graph_structure(self):
        """Heterogeneous graph should have correct node/edge types."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),  # onset with above
            (67, 80, 1.0, 2.0),  # follow after above
        ])
        data = midi_to_hetero_graph(path)

        # Should have note node type with features
        assert data["note"].x.shape == (3, 6)

        # Should have all 4 edge type relations
        for etype in ["onset", "during", "follow", "silence"]:
            assert ("note", etype, "note") in data.edge_types

    def test_hetero_matches_homo_nodes(self):
        """Hetero and homo graphs should have same node features."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.5, 1.5),
            (67, 80, 1.0, 2.0),
        ])
        homo = midi_to_graph(path)
        hetero = midi_to_hetero_graph(path)

        assert torch.allclose(homo.x, hetero["note"].x)

    def test_hetero_edge_count_matches_homo(self):
        """Total edges across all types in hetero should match homo."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),
            (67, 80, 1.0, 2.0),
            (72, 80, 2.0, 3.0),
        ])
        homo = midi_to_graph(path)
        hetero = midi_to_hetero_graph(path)

        total_hetero_edges = sum(
            hetero["note", etype, "note"].edge_index.shape[1]
            for etype in ["onset", "during", "follow", "silence"]
        )
        assert total_hetero_edges == homo.edge_index.shape[1]


class TestScoreGraphPretrainingDataset:
    def test_returns_correct_keys(self):
        """Dataset items should have the expected keys."""
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),
            (67, 80, 1.0, 2.0),
            (72, 80, 2.0, 3.0),
        ])
        graph = midi_to_graph(path)
        ds = ScoreGraphPretrainingDataset([graph])

        item = ds[0]
        assert "x" in item
        assert "edge_index" in item
        assert "pos_edges" in item
        assert "neg_edges" in item

    def test_masked_edges_are_positive_targets(self):
        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),
            (67, 80, 1.0, 2.0),
            (72, 80, 2.0, 3.0),
        ])
        graph = midi_to_graph(path)
        ds = ScoreGraphPretrainingDataset([graph], mask_fraction=0.3)
        item = ds[0]

        # pos_edges should have edges
        assert item["pos_edges"].shape[0] == 2
        assert item["pos_edges"].shape[1] > 0

        # remaining edge_index + pos_edges should equal original edge count
        total = item["edge_index"].shape[1] + item["pos_edges"].shape[1]
        assert total == graph.edge_index.shape[1]

    def test_empty_graphs_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ScoreGraphPretrainingDataset([])


class TestGraphPairCollateFn:
    def test_produces_correct_batch_keys(self):
        """Collated batch should have all keys for GNN finetune step."""
        path1 = _make_midi([
            (60, 80, 0.0, 1.0), (64, 80, 1.0, 2.0),
        ])
        path2 = _make_midi([
            (67, 80, 0.0, 1.0), (72, 80, 1.0, 2.0),
        ])
        g1 = midi_to_graph(path1)
        g2 = midi_to_graph(path2)

        graphs_dict = {"key1": g1, "key2": g2}
        batch = [
            {
                "key_a": "key1",
                "key_b": "key2",
                "labels_a": torch.rand(19),
                "labels_b": torch.rand(19),
                "piece_id": 0,
            }
        ]

        result = graph_pair_collate_fn(batch, graphs_dict)
        assert result is not None
        expected_keys = {
            "x_a", "edge_index_a", "batch_a",
            "x_b", "edge_index_b", "batch_b",
            "labels_a", "labels_b", "piece_ids_a", "piece_ids_b",
        }
        assert set(result.keys()) == expected_keys

    def test_missing_graphs_returns_none(self):
        batch = [
            {
                "key_a": "missing1",
                "key_b": "missing2",
                "labels_a": torch.rand(19),
                "labels_b": torch.rand(19),
                "piece_id": 0,
            }
        ]
        result = graph_pair_collate_fn(batch, {})
        assert result is None


class TestGNNIntegration:
    def test_midi_to_graph_feeds_gnn(self):
        """midi_to_graph output should feed directly into GNNSymbolicEncoder."""
        from model_improvement.symbolic_encoders import GNNSymbolicEncoder

        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),
            (67, 80, 1.0, 2.0),
            (72, 80, 2.0, 3.0),
        ])
        data = midi_to_graph(path)

        model = GNNSymbolicEncoder(
            node_features=6,
            hidden_dim=64,
            num_layers=2,
            num_labels=19,
            heads=2,
        )

        batch_vec = torch.zeros(data.x.shape[0], dtype=torch.long)
        out = model(data.x, data.edge_index, batch_vec)

        assert out["z_symbolic"].shape == (1, 64)
        assert out["scores"].shape == (1, 19)

    def test_midi_to_hetero_graph_feeds_hetero_gnn(self):
        """midi_to_hetero_graph output should feed into GNNHeteroSymbolicEncoder."""
        from model_improvement.symbolic_encoders import GNNHeteroSymbolicEncoder

        path = _make_midi([
            (60, 80, 0.0, 1.0),
            (64, 80, 0.0, 1.0),
            (67, 80, 1.0, 2.0),
            (72, 80, 2.0, 3.0),
        ])
        data = midi_to_hetero_graph(path)

        model = GNNHeteroSymbolicEncoder(
            node_features=6,
            hidden_dim=64,
            num_layers=2,
            num_labels=19,
        )

        x_dict = {"note": data["note"].x}
        edge_index_dict = {}
        for etype in ["onset", "during", "follow", "silence"]:
            edge_index_dict[("note", etype, "note")] = data["note", etype, "note"].edge_index

        batch_vec = torch.zeros(data["note"].x.shape[0], dtype=torch.long)
        out = model(x_dict, edge_index_dict, batch_vec)

        assert out["z_symbolic"].shape == (1, 64)
        assert out["scores"].shape == (1, 19)


class TestHeteroGraphCollateFn:
    def test_produces_correct_batch_keys(self):
        """Collated batch should have all keys for hetero GNN finetune step."""
        path1 = _make_midi([
            (60, 80, 0.0, 1.0), (64, 80, 1.0, 2.0),
        ])
        path2 = _make_midi([
            (67, 80, 0.0, 1.0), (72, 80, 1.0, 2.0),
        ])
        h1 = midi_to_hetero_graph(path1)
        h2 = midi_to_hetero_graph(path2)

        hetero_dict = {"key1": h1, "key2": h2}
        batch = [
            {
                "key_a": "key1",
                "key_b": "key2",
                "labels_a": torch.rand(19),
                "labels_b": torch.rand(19),
                "piece_id": 0,
            }
        ]

        result = hetero_graph_collate_fn(batch, hetero_dict)
        assert result is not None
        expected_keys = {
            "x_dict_a", "edge_index_dict_a", "batch_a",
            "x_dict_b", "edge_index_dict_b", "batch_b",
            "labels_a", "labels_b", "piece_ids_a", "piece_ids_b",
        }
        assert set(result.keys()) == expected_keys

    def test_node_offsets_are_correct(self):
        """Multiple graphs should have correct node offsets in edge indices."""
        path1 = _make_midi([
            (60, 80, 0.0, 1.0), (64, 80, 1.0, 2.0),
        ])
        path2 = _make_midi([
            (67, 80, 0.0, 1.0), (72, 80, 1.0, 2.0),
        ])
        h1 = midi_to_hetero_graph(path1)
        h2 = midi_to_hetero_graph(path2)

        hetero_dict = {"k1": h1, "k2": h2, "k3": h1, "k4": h2}
        batch = [
            {"key_a": "k1", "key_b": "k2",
             "labels_a": torch.rand(19), "labels_b": torch.rand(19), "piece_id": 0},
            {"key_a": "k3", "key_b": "k4",
             "labels_a": torch.rand(19), "labels_b": torch.rand(19), "piece_id": 1},
        ]

        result = hetero_graph_collate_fn(batch, hetero_dict)
        # With 2 items, batch_a should have indices 0 and 1
        assert result["batch_a"].max().item() == 1
        # Total nodes should be sum of both graphs
        n1 = h1["note"].x.shape[0]
        assert result["x_dict_a"]["note"].shape[0] == n1 * 2

    def test_missing_graphs_returns_none(self):
        batch = [
            {
                "key_a": "missing1",
                "key_b": "missing2",
                "labels_a": torch.rand(19),
                "labels_b": torch.rand(19),
                "piece_id": 0,
            }
        ]
        result = hetero_graph_collate_fn(batch, {})
        assert result is None


class TestHeteroPretrainDataset:
    def test_returns_correct_item_format(self):
        path = _make_midi([
            (60, 80, 0.0, 1.0), (64, 80, 0.0, 1.0),
            (67, 80, 1.0, 2.0), (72, 80, 2.0, 3.0),
        ])
        homo = midi_to_graph(path)
        hetero = midi_to_hetero_graph(path)

        ds = HeteroPretrainDataset(
            keys=["k1"],
            homo_graphs={"k1": homo},
            hetero_graphs={"k1": hetero},
        )
        assert len(ds) == 1
        item = ds[0]
        expected_keys = {"x_dict", "edge_index_dict", "pos_edges", "neg_edges"}
        assert set(item.keys()) == expected_keys
        assert "note" in item["x_dict"]
        assert item["pos_edges"].shape[0] == 2

    def test_filters_invalid_keys(self):
        """Keys not in both graph dicts should be filtered out."""
        path = _make_midi([(60, 80, 0.0, 1.0), (64, 80, 1.0, 2.0)])
        homo = midi_to_graph(path)
        hetero = midi_to_hetero_graph(path)

        ds = HeteroPretrainDataset(
            keys=["k1", "missing"],
            homo_graphs={"k1": homo},
            hetero_graphs={"k1": hetero},
        )
        assert len(ds) == 1

    def test_no_valid_keys_raises(self):
        with pytest.raises(ValueError, match="No valid keys"):
            HeteroPretrainDataset(
                keys=["missing"],
                homo_graphs={},
                hetero_graphs={},
            )
