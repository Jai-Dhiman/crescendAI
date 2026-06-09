"""Tests for keys.py -- pure key-resolution helpers."""

import pytest

from exercise_corpus.keys import parse_key_to_pc, transpose_interval, load_passage_key


# --- parse_key_to_pc ---

def test_parse_c_major():
    assert parse_key_to_pc("C major") == 0

def test_parse_c_bare():
    assert parse_key_to_pc("C") == 0

def test_parse_a_minor():
    assert parse_key_to_pc("Am") == 9

def test_parse_a_minor_space():
    assert parse_key_to_pc("A minor") == 9

def test_parse_eb():
    assert parse_key_to_pc("Eb") == 3

def test_parse_eb_minor():
    assert parse_key_to_pc("Ebm") == 3

def test_parse_cs():
    assert parse_key_to_pc("C#") == 1

def test_parse_db():
    assert parse_key_to_pc("Db") == 1

def test_parse_gb():
    assert parse_key_to_pc("Gb") == 6

def test_parse_fs():
    assert parse_key_to_pc("F#") == 6

def test_parse_bb():
    assert parse_key_to_pc("Bb") == 10

def test_parse_g_major():
    assert parse_key_to_pc("G major") == 7

def test_parse_csm():
    assert parse_key_to_pc("C#m") == 1

def test_parse_unknown_raises():
    with pytest.raises(ValueError, match="unparseable"):
        parse_key_to_pc("Q major")

def test_parse_empty_raises():
    with pytest.raises(ValueError, match="unparseable"):
        parse_key_to_pc("")

def test_parse_garbage_raises():
    with pytest.raises(ValueError, match="unparseable"):
        parse_key_to_pc("not a key")


# --- transpose_interval ---

def test_same_key_is_zero():
    assert transpose_interval(0, 0) == 0

def test_same_key_any_pc_is_zero():
    assert transpose_interval(9, 9) == 0

def test_c_to_eb_is_plus_3():
    # C=0 -> Eb=3
    assert transpose_interval(0, 3) == 3

def test_c_to_a_is_minus_3_not_plus_9():
    # C=0 -> A=9; d=9 > 6 so d -= 12 -> -3
    assert transpose_interval(0, 9) == -3

def test_c_to_g_is_plus_7_reduced():
    # C=0 -> G=7; d=7 > 6 so d -= 12 -> -5
    assert transpose_interval(0, 7) == -5

def test_c_to_f_is_plus_5():
    # C=0 -> F=5; d=5 <= 6
    assert transpose_interval(0, 5) == 5

def test_tritone_is_plus_6():
    # C=0 -> F#=6; d=6, convention: +6
    assert transpose_interval(0, 6) == 6

def test_eb_to_c_is_minus_3():
    # Eb=3 -> C=0; d = (0-3)%12 = 9 > 6 -> 9-12 = -3
    assert transpose_interval(3, 0) == -3

def test_range_is_bounded():
    for from_pc in range(12):
        for to_pc in range(12):
            result = transpose_interval(from_pc, to_pc)
            assert -5 <= result <= 6, f"out of range for {from_pc}->{to_pc}: {result}"


# --- load_passage_key ---

from pathlib import Path


def test_load_passage_key_returns_key_from_committed_fixture():
    # Uses the git-committed bach.prelude.bwv_846.json which has key_signature "C major"
    scores_dir = Path(__file__).resolve().parents[3] / "model" / "data" / "scores"
    result = load_passage_key("bach.prelude.bwv_846", scores_dir=scores_dir)
    assert result == "C major"


def test_load_passage_key_raises_for_missing_piece(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="nonexistent_piece"):
        load_passage_key("nonexistent_piece", scores_dir=tmp_path)


def test_load_passage_key_returns_none_when_key_signature_is_null(tmp_path: Path):
    fixture = tmp_path / "no_key_piece.json"
    fixture.write_text('{"piece_id": "no_key_piece", "key_signature": null}')
    result = load_passage_key("no_key_piece", scores_dir=tmp_path)
    assert result is None


def test_load_passage_key_from_test_fixture():
    # Uses a committed fixture at tests/exercise_corpus/fixtures/scores/
    fixtures_scores = Path(__file__).resolve().parent / "fixtures" / "scores"
    result = load_passage_key("test_piece_eb", scores_dir=fixtures_scores)
    assert result == "Eb"
