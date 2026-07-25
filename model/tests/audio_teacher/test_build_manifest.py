"""CSV -> YAML curation: output is loadable by construction."""
from __future__ import annotations

import pytest

from audio_teacher.build_manifest import CurationError, main
from audio_teacher.manifest import load_manifest

_HEADER = "id,axis,population,clip_a,clip_b,degraded,description\n"


def test_csv_curation_round_trips_and_rejects_bad_rows(tmp_path, wav_factory):
    wav_factory("clips/x_a.wav")
    wav_factory("clips/x_b.wav")
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        _HEADER + "x,pedaling,real,clips/x_a.wav,clips/x_b.wav,a,over-pedaled take\n"
    )
    out = tmp_path / "probe_manifest.yaml"
    rc = main(
        [
            "--pairs-csv", str(csv_path),
            "--sample-rate", "16000",
            "--out", str(out),
            "--repo-root", str(tmp_path),
        ]
    )
    assert rc == 0
    manifest = load_manifest(out, repo_root=tmp_path)
    assert [p.pair_id for p in manifest.pairs] == ["x"]
    assert manifest.pairs[0].degraded == "a"
    assert manifest.pairs[0].description == "over-pedaled take"

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        _HEADER + "y,rubato,real,clips/x_a.wav,clips/x_b.wav,a,bad axis\n"
    )
    bad_out = tmp_path / "bad_manifest.yaml"
    with pytest.raises(CurationError) as excinfo:
        main(
            [
                "--pairs-csv", str(bad_csv),
                "--sample-rate", "16000",
                "--out", str(bad_out),
                "--repo-root", str(tmp_path),
            ]
        )
    assert "rubato" in str(excinfo.value)
    assert not bad_out.exists()  # nothing written on invalid input
