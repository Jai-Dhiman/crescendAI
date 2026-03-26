"""Tests for three-way train/val/test split generation."""

import pytest

from model_improvement.splits import (
    generate_t5_splits,
    generate_t1_splits,
    generate_t2_splits,
)


def _make_t5_recordings(pieces=3, buckets=5, per_bucket=6):
    """Helper: create synthetic T5 recording list."""
    recordings = []
    for p in range(pieces):
        for b in range(1, buckets + 1):
            for i in range(per_bucket):
                recordings.append({
                    "video_id": f"piece{p}_bucket{b}_rec{i}",
                    "piece": f"piece_{p}",
                    "skill_bucket": b,
                })
    return recordings


class TestT5Splits:
    def test_stratification_by_piece_and_bucket(self):
        """Every piece+bucket combo appears in every split."""
        recs = _make_t5_recordings(pieces=3, buckets=5, per_bucket=6)
        splits = generate_t5_splits(recs, train=0.8, val=0.1, test=0.1, seed=42)

        for split_name in ("train", "val", "test"):
            split_recs = splits[split_name]
            combos = {(r["piece"], r["skill_bucket"]) for r in split_recs}
            expected = {(f"piece_{p}", b) for p in range(3) for b in range(1, 6)}
            assert combos == expected, f"{split_name} missing combos: {expected - combos}"

    def test_no_recording_leak_across_splits(self):
        """No recording appears in more than one split."""
        recs = _make_t5_recordings()
        splits = generate_t5_splits(recs, train=0.8, val=0.1, test=0.1, seed=42)

        train_ids = {r["video_id"] for r in splits["train"]}
        val_ids = {r["video_id"] for r in splits["val"]}
        test_ids = {r["video_id"] for r in splits["test"]}

        assert train_ids.isdisjoint(val_ids), "train/val overlap"
        assert train_ids.isdisjoint(test_ids), "train/test overlap"
        assert val_ids.isdisjoint(test_ids), "val/test overlap"
        assert len(train_ids) + len(val_ids) + len(test_ids) == len(recs)

    def test_sparse_bucket_raises(self):
        """Piece+bucket with <3 recordings raises ValueError."""
        recs = _make_t5_recordings(pieces=1, buckets=5, per_bucket=6)
        # Remove all but 2 from bucket 3
        recs = [r for r in recs if not (r["skill_bucket"] == 3 and r["video_id"].endswith(("_rec2", "_rec3", "_rec4", "_rec5")))]
        with pytest.raises(ValueError, match="bucket.*fewer than 3"):
            generate_t5_splits(recs, train=0.8, val=0.1, test=0.1, seed=42)


class TestT1Splits:
    def test_stratification_by_piece(self):
        """T1 split is stratified by piece (each piece in both train and test)."""
        records = [
            {"key": f"piece{p}_rec{i}", "piece": f"piece_{p}"}
            for p in range(3) for i in range(10)
        ]
        splits = generate_t1_splits(records, train=0.8, test=0.2, seed=42)

        train_pieces = {r["piece"] for r in splits["train"]}
        test_pieces = {r["piece"] for r in splits["test"]}
        assert train_pieces == test_pieces == {f"piece_{p}" for p in range(3)}


class TestT2Splits:
    def test_holdout_by_round(self):
        """T2 holdout uses entire rounds -- no round appears in both train and test."""
        records = []
        for comp in ("chopin", "cliburn"):
            for round_name in ("prelim", "semifinal", "final"):
                for performer in range(5):
                    records.append({
                        "recording_id": f"{comp}_{round_name}_p{performer}",
                        "competition": comp,
                        "round": round_name,
                        "performer_id": f"{comp}_p{performer}",
                    })
        splits = generate_t2_splits(records, train=0.85, test=0.15, seed=42)

        train_rounds = {(r["competition"], r["round"]) for r in splits["train"]}
        test_rounds = {(r["competition"], r["round"]) for r in splits["test"]}
        assert train_rounds.isdisjoint(test_rounds), "same round in train and test"
