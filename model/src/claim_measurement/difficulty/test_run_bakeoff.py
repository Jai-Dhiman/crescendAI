"""Offline tests for run_bakeoff.py's CLI stage dispatch.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz
from claim_measurement.difficulty.run_bakeoff import main


def _write_fixture_data(data_root: Path):
    manifest = [
        {"seg_id": "a", "key": "A.mid", "grade": 2, "video_id": "x", "midi_name": "mid/A.mid"},
        {"seg_id": "b", "key": "B.mid", "grade": 5, "video_id": "y", "midi_name": "mid/B.mid"},
    ]
    labels = {"A.mid": {"composer": "Bach"}, "B.mid": {"composer": "Czerny"}}
    (data_root / "results" / "amt_gap_curve").mkdir(parents=True)
    (data_root / "raw" / "psyllabus").mkdir(parents=True)
    (data_root / "results" / "amt_gap_curve" / "manifest.json").write_text(json.dumps(manifest))
    (data_root / "raw" / "psyllabus" / "new_clean_data.json").write_text(json.dumps(labels))
    mid_dir = data_root / "results" / "amt_gap_curve" / "transkun_mid"
    mid_dir.mkdir(parents=True)
    (mid_dir / "a.mid").write_bytes(b"")
    (mid_dir / "b.mid").write_bytes(b"")


def test_sample_stage_writes_sample_manifest(tmp_path):
    _write_fixture_data(tmp_path)

    exit_code = main(["--stage", "sample", "--data-root", str(tmp_path), "--target-n", "2"])

    assert exit_code == 0
    out = json.loads((tmp_path / "results" / "bakeoff" / "sample_manifest.json").read_text())
    assert {e["seg_id"] for e in out} == {"a", "b"}


def test_eval_stage_reports_empty_dict_when_emb_dir_missing(tmp_path, capsys):
    # No extraction has run yet -- results/bakeoff/emb/ does not exist. The
    # eval stage must report an empty result set, not raise.
    exit_code = main(["--stage", "eval", "--data-root", str(tmp_path)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {}


def test_eval_stage_skips_backbone_dir_with_no_npz_files(tmp_path, capsys):
    # A backbone subdirectory exists (e.g. created but extraction produced
    # nothing) but holds no .npz files: it must be skipped, not crash the
    # npz-glob/stack logic.
    (tmp_path / "results" / "bakeoff" / "emb" / "aria").mkdir(parents=True)

    exit_code = main(["--stage", "eval", "--data-root", str(tmp_path)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {}


def test_eval_stage_reports_per_backbone_per_pooling_tau_c(tmp_path, capsys):
    rng = np.random.default_rng(0)
    emb_dir = tmp_path / "results" / "bakeoff" / "emb" / "aria"
    for i in range(12):
        write_embedding_npz(
            emb_dir / f"piece_{i}.npz",
            {"embedding": rng.random(4).astype(np.float32)},
            grade=i % 6,
            composer_id=i % 6,
        )

    exit_code = main(["--stage", "eval", "--data-root", str(tmp_path)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert "aria" in printed
    assert set(printed["aria"]["embedding"]) == {"mean", "std", "n_seeds"}
