"""Offline test of moonbeam_extract_script's CLI wiring, via an injected fake
loader_factory -- never touches the real moonbeam fork or checkpoint.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np

from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.moonbeam_extract_script import main


def test_main_wires_injected_loader_into_extraction(tmp_path):
    sample_manifest = tmp_path / "sample_manifest.json"
    sample_manifest.write_text(json.dumps([
        {"seg_id": "a", "key": "A.mid", "grade": 2, "composer": "Bach"},
    ]))
    out_dir = tmp_path / "emb"
    composer_index = tmp_path / "composer_index.json"

    seen = {}

    def fake_loader_factory(checkpoint_path, repo_root, model_config):
        seen["repo_root"] = repo_root
        seen["model_config"] = model_config
        return lambda midi_path: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    exit_code = main(
        [
            "--checkpoint", str(tmp_path / "fake.pt"),
            "--sample-manifest", str(sample_manifest),
            "--midi-dir", str(tmp_path / "mid"),
            "--out-dir", str(out_dir),
            "--composer-index", str(composer_index),
            "--repo-root", str(tmp_path / "repo"),
            "--model-config", str(tmp_path / "repo" / "model_config.json"),
        ],
        loader_factory=fake_loader_factory,
    )

    assert seen["repo_root"] == tmp_path / "repo"
    assert seen["model_config"] == tmp_path / "repo" / "model_config.json"

    assert exit_code == 0
    record = read_embedding_npz(out_dir / "a.npz")
    np.testing.assert_allclose(record.embeddings["mean_pool"], [2.0, 3.0])
    np.testing.assert_allclose(record.embeddings["last_token"], [3.0, 4.0])
