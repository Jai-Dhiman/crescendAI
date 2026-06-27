# model/tests/exercise_corpus/test_sources.py
import tomllib
from pathlib import Path

# CORRECTED: parents[2] from model/tests/exercise_corpus/ goes to model/
SOURCES_PATH = Path(__file__).parents[2] / "src" / "exercise_corpus" / "sources.toml"


def test_sources_toml_exists():
    assert SOURCES_PATH.exists(), f"sources.toml not found at {SOURCES_PATH}"


# The #17 Mutopia core (6) + the #49 EXERCISE-BOOK expansion. The prior #49
# wave's 350 whole-movement sonata primitives (beethoven/mozart/scarlatti/
# haydn/joplin/chopin-mazurkas) were dropped -- they are repertoire, not drills,
# and belong in the piece-ID library (#96). The exercise corpus is now study/
# etude BOOKS whose published numbered unit is the drill.
_CORE_SOURCES = {"hanon", "bach", "czerny", "burgmuller", "chopin", "satie"}
_EXERCISE_BOOK_SOURCES = {
    "czerny_op821",
    "chopin_etudes",
    "clementi_preludes",
}


def test_sources_has_all_entries():
    with open(SOURCES_PATH, "rb") as f:
        data = tomllib.load(f)
    assert "sources" in data
    assert len(data["sources"]) == len(_CORE_SOURCES) + len(_EXERCISE_BOOK_SOURCES)


def test_each_source_has_required_keys():
    with open(SOURCES_PATH, "rb") as f:
        data = tomllib.load(f)
    required_keys = {"name", "license", "musicxml_path"}
    for source in data["sources"]:
        missing = required_keys - set(source.keys())
        assert not missing, f"Source {source.get('name', '?')} missing keys: {missing}"


def test_source_names_match_expected():
    with open(SOURCES_PATH, "rb") as f:
        data = tomllib.load(f)
    names = {s["name"] for s in data["sources"]}
    assert names == _CORE_SOURCES | _EXERCISE_BOOK_SOURCES


def test_exercise_book_sources_carry_coarse_tags():
    # The #49 exercise-book sources declare coarse source-level dimension tags
    # (per-primitive technique_tags carry the authoritative tagging). Every
    # declared dimension must be one of the canonical 6.
    from exercise_corpus.tags import DIMENSIONS

    with open(SOURCES_PATH, "rb") as f:
        data = tomllib.load(f)
    for source in data["sources"]:
        if source["name"] not in _EXERCISE_BOOK_SOURCES:
            continue
        dims = source.get("dimensions", [])
        assert dims, f"exercise-book source {source['name']} missing source-level dimensions"
        for d in dims:
            assert d in DIMENSIONS, f"{source['name']}: unknown dimension {d!r}"
