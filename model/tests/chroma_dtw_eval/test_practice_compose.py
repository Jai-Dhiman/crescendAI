import numpy as np

from chroma_dtw_eval.practice_compose import compose_batch


def test_compose_batch_produces_all_four_patterns_with_known_truth():
    source = np.random.RandomState(1).rand(12, 4000).astype(np.float32)
    source /= np.linalg.norm(source, axis=0, keepdims=True) + 1e-9
    batch = compose_batch(source, n_per_pattern=2, chunk_len_frames=750, seed=11)
    kinds = {seq.pattern for seq in batch}
    assert kinds == {"repeat", "restart", "jump", "partial"}
    for seq in batch:
        assert seq.chroma.shape[0] == 12
        assert seq.chroma.shape[1] == 750
        assert len(seq.stitch_points) >= 1
        for synth_f, src_f in seq.stitch_points:
            assert 0 <= synth_f < 750
            assert 0 <= src_f < source.shape[1]
