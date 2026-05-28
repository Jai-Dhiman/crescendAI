# model/tests/exercise_corpus/test_fixtures.py
import partitura
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_hanon_fixture_has_three_parts():
    score = partitura.load_musicxml(str(FIXTURES / "hanon_3ex.xml"))
    assert len(score.parts) == 3


def test_czerny_fixture_has_three_parts():
    score = partitura.load_musicxml(str(FIXTURES / "czerny_3ex.xml"))
    assert len(score.parts) == 3


def test_burgmuller_fixture_has_three_parts():
    score = partitura.load_musicxml(str(FIXTURES / "burgmuller_3ex.xml"))
    assert len(score.parts) == 3


def test_package_importable():
    import exercise_corpus  # noqa: F401
