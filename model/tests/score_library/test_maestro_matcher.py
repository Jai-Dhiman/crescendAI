"""Tests for MAESTRO-to-ASAP fuzzy matching engine."""
import pytest

from src.score_library.maestro_matcher import extract_composer_last_name, match_composer


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
