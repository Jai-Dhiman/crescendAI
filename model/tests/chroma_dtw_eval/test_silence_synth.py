import numpy as np

from chroma_dtw_eval.silence_synth import generate_silence_chunks


def test_generate_silence_chunks_yields_low_rms_with_mixed_kinds():
    chunks = generate_silence_chunks(n=10, sr=24000, chunk_len_s=15.0, seed=7)
    assert len(chunks) == 10
    kinds = {c.kind for c in chunks}
    assert "zero" in kinds
    assert "low_noise" in kinds
    for c in chunks:
        rms = float(np.sqrt(np.mean(c.waveform.astype(np.float64) ** 2)))
        assert rms < 0.02, f"chunk {c.kind} rms {rms} too high"
        assert c.waveform.dtype == np.float32
        assert c.waveform.shape == (24000 * 15,)
