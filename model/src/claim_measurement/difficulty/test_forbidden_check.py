"""Tests for the #171 forbidden piece/composer intersection check.

The fixtures mirror the real artifact shapes: head_manifest rows carry an
anonymized `composer_id` that must equal the index into the sorted distinct
joined composer names, which is what `check_composer_ids` verifies.
"""
from __future__ import annotations

import json

import pytest

from claim_measurement.difficulty.forbidden_check import (
    build_trained_pool,
    check_composer_ids,
    format_report,
    full_name_from_bio_link,
    main,
    match_composers,
    match_pieces,
    normalize,
    run_check,
    surname_of,
)

# 'Bach J.S.' sorts first, then 'Chopin F.', then 'Faure G.' -- composer_id
# follows that order, as push_train_dataset.build_head_manifest assigns it.
MANIFEST = [
    {"seg_id": "Bach_JS_Invention_1", "key": "Bach J.S.Invention 1"},
    {"seg_id": "Bach_JS_Invention_2", "key": "Bach J.S.Invention 2"},
    {"seg_id": "Chopin_F_Nocturne_op_9_no_2", "key": "Chopin F.Nocturne op 9 no 2"},
    {"seg_id": "Faure_G_Barcarolle_9", "key": "Faure G.Barcarolle 9"},
]
WIKI = "http://en.wikipedia.org/wiki/"
BACH_LINK = WIKI + "Johann_Sebastian_Bach"
CHOPIN_LINK = WIKI + "Fr%C3%A9d%C3%A9ric_Chopin"
PS_LINK = "https://pianosyllabus.com/x-detail.php?ref=1"
LABELS = {
    "Bach J.S.Invention 1": {
        "composer": "Bach J.S.", "PS_title": "Invention 1",
        "composer_bio_link": BACH_LINK},
    "Bach J.S.Invention 2": {
        "composer": "Bach J.S.", "PS_title": "Invention 2",
        "composer_bio_link": BACH_LINK},
    "Chopin F.Nocturne op 9 no 2": {
        "composer": "Chopin F.", "PS_title": "Nocturne op 9 no 2",
        "composer_bio_link": CHOPIN_LINK},
    "Faure G.Barcarolle 9": {
        "composer": "Faure G.", "PS_title": "Barcarolle 9",
        "composer_bio_link": PS_LINK},
}
HEAD = [
    {"seg_id": "Bach_JS_Invention_1", "grade": 4, "composer_id": 0},
    {"seg_id": "Bach_JS_Invention_2", "grade": 4, "composer_id": 0},
    {"seg_id": "Chopin_F_Nocturne_op_9_no_2", "grade": 7, "composer_id": 1},
    {"seg_id": "Faure_G_Barcarolle_9", "grade": 7, "composer_id": 2},
]
FOLD_PLANS = [
    {"fold": 0, "test_seg_ids": ["Faure_G_Barcarolle_9"],
     "train_seg_ids": ["Bach_JS_Invention_1", "Chopin_F_Nocturne_op_9_no_2"],
     "val_seg_ids": ["Bach_JS_Invention_2"]},
    {"fold": 1, "test_seg_ids": ["Chopin_F_Nocturne_op_9_no_2"],
     "train_seg_ids": ["Bach_JS_Invention_1", "Faure_G_Barcarolle_9"],
     "val_seg_ids": []},
    {"fold": 99, "test_seg_ids": [],
     "train_seg_ids": [m["seg_id"] for m in MANIFEST], "val_seg_ids": []},
]


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Frédéric Chopin") == "frederic chopin"
    assert normalize("Chopin, F.") == "chopin f"
    assert normalize("  Bach   J.S. ") == "bach j s"


def test_surname_is_the_leading_token():
    assert surname_of("Chopin F.") == "chopin"
    assert surname_of("Bach J.S.") == "bach"
    assert surname_of("") == ""


def test_full_name_comes_only_from_wikipedia_links():
    assert full_name_from_bio_link(CHOPIN_LINK) == "frederic chopin"
    assert full_name_from_bio_link(PS_LINK) is None
    assert full_name_from_bio_link(None) is None


def test_build_trained_pool_joins_composer_and_title():
    pool = build_trained_pool(HEAD, MANIFEST, LABELS)
    assert len(pool) == 4
    chopin = next(p for p in pool if p.composer == "Chopin F.")
    assert chopin.title == "Nocturne op 9 no 2"
    assert "frederic chopin" in chopin.aliases


def test_build_trained_pool_refuses_an_unresolvable_piece():
    """A trained seg_id with no composer must abort the run, never be skipped:
    an unresolved piece would silently read as 'not forbidden'."""
    head = HEAD + [{"seg_id": "Ghost_Piece", "grade": 1, "composer_id": 3}]
    with pytest.raises(ValueError, match="missing from the manifest"):
        build_trained_pool(head, MANIFEST, LABELS)

    labels = dict(LABELS)
    labels["Faure G.Barcarolle 9"] = {"composer": "  ", "PS_title": "Barcarolle 9"}
    with pytest.raises(ValueError, match="no composer in the labels"):
        build_trained_pool(HEAD, MANIFEST, labels)


def test_check_composer_ids_accepts_the_real_partition():
    check_composer_ids(build_trained_pool(HEAD, MANIFEST, LABELS), HEAD)


def test_check_composer_ids_rejects_a_drifted_join():
    """If the labels join ever stops reproducing the trained composer_id
    grouping, the check must abort rather than report a verdict from a
    different set of composers than the training run saw."""
    pool = build_trained_pool(HEAD, MANIFEST, LABELS)
    drifted = [dict(h) for h in HEAD]
    drifted[1]["composer_id"] = 3  # split Bach's two pieces across two ids
    with pytest.raises(ValueError, match="composer join drifted"):
        check_composer_ids(pool, drifted)


@pytest.mark.parametrize("entry", ["Chopin", "Frédéric Chopin", "Chopin, Frederic",
                                    "chopin f."])
def test_composer_matching_is_recall_first_across_spellings(entry):
    pool = build_trained_pool(HEAD, MANIFEST, LABELS)
    matches, unmatched = match_composers(pool, [entry])
    assert matches[entry] == ["Chopin F."]
    assert unmatched == []


def test_unmatched_forbidden_composers_are_reported_not_swallowed():
    pool = build_trained_pool(HEAD, MANIFEST, LABELS)
    matches, unmatched = match_composers(pool, ["Ligeti"])
    assert matches == {}
    assert unmatched == ["Ligeti"]


def test_piece_matching_uses_exact_normalized_titles():
    pool = build_trained_pool(HEAD, MANIFEST, LABELS)
    matches, unmatched = match_pieces(pool, ["Nocturne op 9 no 2", "Nocturne"])
    assert matches == {"Nocturne op 9 no 2": ["Chopin_F_Nocturne_op_9_no_2"]}
    assert unmatched == ["Nocturne"]


def test_clean_list_is_compliant():
    report = run_check(HEAD, MANIFEST, LABELS, FOLD_PLANS,
                       {"composers": ["Ligeti"], "pieces": []})
    assert report["compliant"] is True
    assert report["n_offending_pieces"] == 0
    assert report["affected_plans"] == {}
    assert report["composer_unmatched"] == ["Ligeti"]


def test_a_forbidden_composer_condemns_every_piece_by_that_composer():
    """The point of the composer clause: one blacklisted name removes all of
    that composer's pieces, and names every adapter trained on any of them."""
    report = run_check(HEAD, MANIFEST, LABELS, FOLD_PLANS,
                       {"composers": ["Johann Sebastian Bach"], "pieces": []})
    assert report["compliant"] is False
    assert report["offending_seg_ids"] == ["Bach_JS_Invention_1", "Bach_JS_Invention_2"]
    # fold 0 trains on Invention 1 and validates on Invention 2; fold 1 trains
    # on Invention 1; the all-data plan holds both.
    assert report["affected_plans"] == {0: 2, 1: 1, 99: 2}


def test_a_forbidden_piece_condemns_only_that_piece():
    report = run_check(HEAD, MANIFEST, LABELS, FOLD_PLANS,
                       {"composers": [], "pieces": ["Barcarolle 9"]})
    assert report["offending_seg_ids"] == ["Faure_G_Barcarolle_9"]
    # fold 0 holds Faure out as TEST, so no fold-0 adapter saw it.
    assert report["affected_plans"] == {1: 1, 99: 1}


def test_format_report_names_the_offenders():
    report = run_check(HEAD, MANIFEST, LABELS, FOLD_PLANS,
                       {"composers": ["Chopin"], "pieces": []})
    text = format_report(report)
    assert "NON-COMPLIANT" in text
    assert "'Chopin' -> catalog ['Chopin F.']" in text
    assert "all-data submission plan" in text


def test_main_exit_code_is_the_verdict(tmp_path, capsys):
    """Exit 1 on a violation so the check can gate a freeze from a shell."""
    def write(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return p

    head = write("head.json", HEAD)
    plans = write("plans.json", FOLD_PLANS)
    manifest = write("manifest.json", MANIFEST)
    labels = write("labels.json", LABELS)
    argv = ["--head-manifest", str(head), "--fold-plans", str(plans),
            "--manifest", str(manifest), "--labels", str(labels)]

    clean = write("clean.json", {"composers": [], "pieces": []})
    assert main(argv + ["--forbidden", str(clean)]) == 0

    dirty = write("dirty.json", {"composers": ["Chopin"], "pieces": []})
    out_path = tmp_path / "report.json"
    assert main(argv + ["--forbidden", str(dirty), "--json-out", str(out_path)]) == 1
    assert json.loads(out_path.read_text())["compliant"] is False
    assert "NON-COMPLIANT" in capsys.readouterr().out
