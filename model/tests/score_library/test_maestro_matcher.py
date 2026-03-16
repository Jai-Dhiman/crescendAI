"""Tests for MAESTRO-to-ASAP fuzzy matching engine."""
import pytest

from src.score_library.maestro_matcher import (
    extract_composer_last_name,
    match_composer,
    normalize_title,
    detect_multi_piece,
)


class TestExtractComposerLastName:
    def test_simple_name(self):
        assert extract_composer_last_name("Frédéric Chopin") == "chopin"

    def test_van_prefix(self):
        assert extract_composer_last_name("Ludwig van Beethoven") == "beethoven"

    def test_bach_full(self):
        assert extract_composer_last_name("Johann Sebastian Bach") == "bach"

    def test_arrangement_slash(self):
        # "Franz Schubert / Franz Liszt" -> primary composer is Schubert
        assert extract_composer_last_name("Franz Schubert / Franz Liszt") == "schubert"

    def test_accented_characters(self):
        assert extract_composer_last_name("César Franck") == "franck"

    def test_single_name(self):
        assert extract_composer_last_name("Rachmaninoff") == "rachmaninoff"


class TestMatchComposer:
    """Match MAESTRO composer to ASAP composer prefix."""

    ASAP_COMPOSERS = [
        "bach", "beethoven", "chopin", "debussy", "haydn", "liszt",
        "mozart", "rachmaninoff", "schubert", "schumann", "scriabin",
        "brahms", "ravel", "prokofiev", "balakirev", "glinka",
    ]

    def test_chopin_match(self):
        assert match_composer("Frédéric Chopin", self.ASAP_COMPOSERS) == "chopin"

    def test_beethoven_match(self):
        assert match_composer("Ludwig van Beethoven", self.ASAP_COMPOSERS) == "beethoven"

    def test_bach_match(self):
        assert match_composer("Johann Sebastian Bach", self.ASAP_COMPOSERS) == "bach"

    def test_arrangement_uses_primary(self):
        assert match_composer("Franz Schubert / Franz Liszt", self.ASAP_COMPOSERS) == "schubert"

    def test_scriabin_match(self):
        assert match_composer("Alexander Scriabin", self.ASAP_COMPOSERS) == "scriabin"

    def test_no_match_returns_none(self):
        assert match_composer("Alban Berg", self.ASAP_COMPOSERS) is None

    def test_balakirev_arrangement(self):
        # "Mikhail Glinka / Mily Balakirev" -> primary is Glinka
        assert match_composer("Mikhail Glinka / Mily Balakirev", self.ASAP_COMPOSERS) == "glinka"

    def test_rachmaninoff_match(self):
        assert match_composer("Sergei Rachmaninoff", self.ASAP_COMPOSERS) == "rachmaninoff"


class TestDetectMultiPiece:
    def test_number_range(self):
        assert detect_multi_piece("24 Preludes Op. 11, No. 13-24") is True

    def test_nos_range(self):
        assert detect_multi_piece("Nos. 1-6") is True

    def test_books(self):
        assert detect_multi_piece("Well-Tempered Clavier Books I & II") is True

    def test_complete(self):
        assert detect_multi_piece("Complete Etudes") is True

    def test_single_piece(self):
        assert detect_multi_piece("Ballade No. 1 in G Minor") is False

    def test_single_number(self):
        assert detect_multi_piece("Etude Op. 10 No. 3") is False


class TestNormalizeTitle:
    """Normalize MAESTRO titles and ASAP title components to comparable tokens."""

    def test_opus_normalization(self):
        assert "op_10" in normalize_title("Etudes Op. 10")

    def test_opus_no_space(self):
        assert "op_10" in normalize_title("Etudes op.10")

    def test_opus_word(self):
        assert "op_10" in normalize_title("Etudes Opus 10")

    def test_number_extraction(self):
        assert "3" in normalize_title("No. 3 in E major")

    def test_number_nr(self):
        assert "3" in normalize_title("Nr. 3")

    def test_bwv_catalog(self):
        assert "bwv_846" in normalize_title("Prelude BWV 846")

    def test_k_catalog(self):
        assert "k_331" in normalize_title("Sonata K. 331")

    def test_d_catalog(self):
        assert "d_899" in normalize_title("Impromptu D. 899")

    def test_strip_common_prefixes(self):
        result = normalize_title("Piano Sonata No. 23 in F minor")
        assert "piano" not in result
        assert "sonata" in result

    def test_strip_key_signatures(self):
        result = normalize_title("Ballade No. 1 in G Minor")
        # Key info (in G Minor) should be stripped
        assert "minor" not in result

    def test_accent_stripping(self):
        result = normalize_title("Étude")
        assert "etude" in result

    def test_etude_singular(self):
        result = normalize_title("Etudes Op. 10")
        assert "etude" in result

    def test_returns_sorted_tokens(self):
        # Normalization returns a sorted list of tokens for stable comparison
        result = normalize_title("Sonata No. 23")
        assert isinstance(result, list)
