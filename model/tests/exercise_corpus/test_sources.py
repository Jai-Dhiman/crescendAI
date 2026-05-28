# model/tests/exercise_corpus/test_sources.py
import tomllib
from pathlib import Path

# CORRECTED: parents[2] from model/tests/exercise_corpus/ goes to model/
SOURCES_PATH = Path(__file__).parents[2] / "src" / "exercise_corpus" / "sources.toml"


def test_sources_toml_exists():
    assert SOURCES_PATH.exists(), f"sources.toml not found at {SOURCES_PATH}"


def test_sources_has_three_entries():
    with open(SOURCES_PATH, "rb") as f:
        data = tomllib.load(f)
    assert "sources" in data
    assert len(data["sources"]) == 3


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
    assert names == {"hanon", "czerny", "burgmuller"}
