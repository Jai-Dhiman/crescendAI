# model/tests/follower_eval/test_validate_tool.py
"""Unit tests for the validator's SCORE-RESOLUTION path (issue #133).

The corpus folder labels are known-wrong, so which score a clip gets validated
against is the correctness-critical decision in this tool: hand the human the
wrong score and they will record a follower failure that never happened. These
tests pin that resolution (piece-ID map -> score, abstain -> flagged label
fallback, cache keyed by score) on constructed inputs; the follower itself and
the canvas rendering are covered elsewhere.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from follower_eval import validate_tool as vt


def _write_piece_id(tmp_path, rows):
    p = tmp_path / "_piece_id.json"
    p.write_text(json.dumps(rows))
    return p


def _bundle(tmp_path, piece, vid, wav_name="a.wav"):
    """A minimal bundle + its WAV, as list_clips requires both to exist."""
    wav = tmp_path / wav_name
    wav.write_bytes(b"RIFF")
    d = tmp_path / "bundles" / piece
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{vid}.json").write_text(json.dumps({"audio_path": str(wav), "title": "t"}))
    return tmp_path / "bundles"


def test_load_piece_id_map_drops_abstains(tmp_path):
    p = _write_piece_id(
        tmp_path,
        [
            {
                "piece_folder": "fantaisie_impromptu",
                "video_id": "x",
                "decision": "chopin.etudes_op_25.5",
            },
            {"piece_folder": "fur_elise", "video_id": "y", "decision": "ABSTAIN"},
        ],
    )
    m = vt.load_piece_id_map(p)
    assert m == {"fantaisie_impromptu/x": "chopin.etudes_op_25.5"}


def test_load_piece_id_map_missing_file_is_loud(tmp_path):
    # never silently fall back to the wrong-label path
    with pytest.raises(vt.ValidateToolError, match="piece_id"):
        vt.load_piece_id_map(tmp_path / "nope.json")


def test_resolve_prefers_identified_score_over_label():
    sid, src = vt.resolve_score_id(
        "fantaisie_impromptu", "x", {"fantaisie_impromptu/x": "chopin.etudes_op_25.5"}
    )
    assert (sid, src) == ("chopin.etudes_op_25.5", "piece_id")


def test_resolve_falls_back_to_label_and_flags_it():
    # abstained clip: still validatable, but the source must say it is unverified
    sid, src = vt.resolve_score_id("bach_prelude_c_wtc1", "y", {})
    assert sid == "bach.prelude.bwv_846"  # folder label != score_id
    assert src == "label"


def test_resolve_unknown_piece_without_id_is_loud():
    with pytest.raises(vt.ValidateToolError):
        vt.resolve_score_id("not_a_rep_piece", "y", {})


def test_list_clips_attaches_resolved_score_and_relabel_flag(tmp_path):
    bundles = _bundle(tmp_path, "fantaisie_impromptu", "x")
    subset = tmp_path / "subset.json"
    subset.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "piece": "fantaisie_impromptu",
                        "video_id": "x",
                        "v1_confidence": 0.1,
                    }
                ]
            }
        )
    )
    clips = vt.list_clips(
        subset,
        bundles,
        use_all=False,
        pieces=None,
        id_map={"fantaisie_impromptu/x": "chopin.etudes_op_25.5"},
    )
    assert clips[0]["score_id"] == "chopin.etudes_op_25.5"
    assert clips[0]["score_source"] == "piece_id"
    assert clips[0]["relabeled"] is True


def test_list_clips_confirmed_label_is_not_relabeled(tmp_path):
    # piece-ID agreeing with the folder label must NOT be reported as a re-label
    bundles = _bundle(tmp_path, "bach_prelude_c_wtc1", "w")
    subset = tmp_path / "subset.json"
    subset.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "piece": "bach_prelude_c_wtc1",
                        "video_id": "w",
                        "v1_confidence": 0.9,
                    }
                ]
            }
        )
    )
    clips = vt.list_clips(
        subset,
        bundles,
        use_all=False,
        pieces=None,
        id_map={"bach_prelude_c_wtc1/w": "bach.prelude.bwv_846"},
    )
    assert clips[0]["relabeled"] is False


def test_view_cache_key_separates_scores(tmp_path):
    # a re-labeled clip must not be served the view built against its old score
    a = vt._view_cache_path(
        tmp_path, "fantaisie_impromptu", "x", "chopin.fantaisie_impromptu"
    )
    b = vt._view_cache_path(
        tmp_path, "fantaisie_impromptu", "x", "chopin.etudes_op_25.5"
    )
    assert a != b


def test_generate_html_surfaces_unverified_scores(tmp_path):
    clips = [
        {
            "piece": "fur_elise",
            "video_id": "y",
            "title": None,
            "v1_confidence": 0.2,
            "existing": False,
            "score_id": "beethoven.fur_elise",
            "score_source": "label",
            "relabeled": False,
        }
    ]
    html = vt.generate_html(clips)
    assert "SCORE UNVERIFIED" in html
    assert '"score_source": "label"' in html or "'score_source': 'label'" in html


def test_build_clip_view_reports_median_confidence(tmp_path, monkeypatch):
    score_id = "bach.prelude.bwv_846"
    (tmp_path / f"{score_id}.json").write_text("{}")
    monkeypatch.setattr(
        vt,
        "load_score",
        lambda _path: ([SimpleNamespace(position=0.0, pitch=60)], [], 1.0),
    )
    monkeypatch.setattr(
        vt,
        "load_bundle_notes",
        lambda _path: [SimpleNamespace(onset=0.0, pitch=60)],
    )
    matches = [
        SimpleNamespace(perf_time=0.0, score_position=0.0, confidence=0.1),
        SimpleNamespace(perf_time=1.0, score_position=1.0, confidence=0.2),
        SimpleNamespace(perf_time=2.0, score_position=2.0, confidence=0.9),
    ]
    monkeypatch.setattr(
        vt,
        "follow_hmm",
        lambda *_args, **_kwargs: SimpleNamespace(
            matches=matches,
            transpose_semitones=0,
        ),
    )

    view = vt.build_clip_view(
        "bach_prelude_c_wtc1", tmp_path / "clip.json", tmp_path, score_id
    )

    assert view["median_confidence"] == 0.2


def test_save_validation_persists_resolved_score_confidence(tmp_path):
    path = vt.save_validation(
        tmp_path,
        {
            "piece": "fantaisie_impromptu",
            "video_id": "x",
            "verdict": "tracked",
            "wrong_spans": [],
            "score_id": "chopin.etudes_op_25.5",
            "score_source": "piece_id",
            "follower_confidence": 0.84,
        },
    )

    saved = json.loads(path.read_text())
    assert saved["score_id"] == "chopin.etudes_op_25.5"
    assert saved["score_source"] == "piece_id"
    assert saved["follower_confidence"] == 0.84


def test_save_validation_requires_score_provenance(tmp_path):
    with pytest.raises(vt.ValidateToolError, match="score provenance"):
        vt.save_validation(
            tmp_path,
            {"piece": "fur_elise", "video_id": "y", "verdict": "tracked"},
        )


def test_save_validation_rejects_path_traversal(tmp_path):
    with pytest.raises(vt.ValidateToolError, match="unsafe clip identifier"):
        vt.save_validation(
            tmp_path,
            {
                "piece": "../outside",
                "video_id": "x",
                "verdict": "tracked",
                "score_id": "score",
                "score_source": "piece_id",
                "follower_confidence": 0.8,
            },
        )
