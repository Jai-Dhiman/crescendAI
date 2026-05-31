from chroma_dtw_eval.chunk_sampler import PieceSpec, sample_chunks


def test_sample_chunks_is_deterministic_and_stratified():
    pieces = [
        PieceSpec(piece_id="p1", duration_s=300.0),
        PieceSpec(piece_id="p2", duration_s=600.0),
    ]
    a = sample_chunks(pieces, n_per_piece=10, chunk_len_s=15.0, seed=42)
    b = sample_chunks(pieces, n_per_piece=10, chunk_len_s=15.0, seed=42)
    assert [c.start_s for c in a] == [c.start_s for c in b]
    assert [c.position_bucket for c in a] == [c.position_bucket for c in b]

    buckets_p1 = {c.position_bucket for c in a if c.piece_id == "p1"}
    assert buckets_p1 == {"intro", "early", "middle", "late", "cadence"}
    for c in a:
        assert 0.0 <= c.start_s <= c.piece_duration_s - c.chunk_len_s + 1e-6
