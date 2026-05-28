# model/tests/exercise_corpus/test_fixtures.py
import partitura
from partitura.score import Repeat
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_hanon_fixture_is_single_part_with_three_repeats():
    # Real method-book MusicXML is one part; exercises are delimited by Repeat
    # barlines, not by separate parts.
    score = partitura.load_musicxml(str(FIXTURES / "hanon_3ex.xml"))
    assert len(score.parts) == 1
    repeats = list(score.parts[0].iter_all(Repeat))
    assert len(repeats) == 3


def test_czerny_fixture_is_single_part():
    score = partitura.load_musicxml(str(FIXTURES / "czerny_1ex.xml"))
    assert len(score.parts) == 1


def test_burgmuller_fixture_is_single_part():
    score = partitura.load_musicxml(str(FIXTURES / "burgmuller_1ex.xml"))
    assert len(score.parts) == 1


def test_package_importable():
    import exercise_corpus  # noqa: F401
