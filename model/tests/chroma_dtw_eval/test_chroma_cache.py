import numpy as np
import soundfile as sf

from chroma_dtw_eval.chroma_cache import ChromaParams, get_chroma


def test_get_chroma_caches_after_first_call(tmp_path):
    sr = 24000
    y = np.random.RandomState(0).randn(sr * 2).astype(np.float32) * 0.1
    audio = tmp_path / "a.wav"
    sf.write(audio, y, sr)
    cache_root = tmp_path / "cache"
    params = ChromaParams(target_frame_rate_hz=50.0, sr=sr)

    first = get_chroma(audio, params, cache_root=cache_root)
    cached_files = list(cache_root.rglob("*.bin"))
    assert len(cached_files) == 1, f"expected 1 cache file, got {cached_files}"
    mtime = cached_files[0].stat().st_mtime_ns

    second = get_chroma(audio, params, cache_root=cache_root)
    assert cached_files[0].stat().st_mtime_ns == mtime, "cache file was rewritten"
    assert np.array_equal(first.data, second.data)
    assert first.data.shape[0] == 12
    assert first.data.dtype == np.float32
