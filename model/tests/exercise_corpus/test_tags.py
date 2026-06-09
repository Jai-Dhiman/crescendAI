"""Tests for tags.py -- the editorial technique-tag layer.

load_tags reads a version-controlled technique_tags.toml and validates it against
the catalog: every dimension label must be one of the canonical 6, and every
tagged primitive_id must exist in the catalog. Tests use tmp_path TOML fixtures
and an explicit known_primitive_ids set, so no catalog DB or Aria weights are
required.
"""

from pathlib import Path

import pytest

from exercise_corpus.tags import TagSet, load_tags


def _write_toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "technique_tags.toml"
    p.write_text(body)
    return p


def test_load_tags_reads_valid_table(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[hanon_001]
dimensions = ["articulation", "timing"]
techniques = ["finger_independence", "evenness"]

[burgmuller_001]
dimensions = ["phrasing", "interpretation"]
""",
    )
    tags = load_tags(toml, known_primitive_ids={"hanon_001", "burgmuller_001"})

    assert set(tags) == {"hanon_001", "burgmuller_001"}
    assert isinstance(tags["hanon_001"], TagSet)
    assert tags["hanon_001"].dimensions == frozenset({"articulation", "timing"})
    assert tags["hanon_001"].techniques == frozenset(
        {"finger_independence", "evenness"}
    )
    # techniques key may be omitted -> empty frozenset
    assert tags["burgmuller_001"].dimensions == frozenset(
        {"phrasing", "interpretation"}
    )
    assert tags["burgmuller_001"].techniques == frozenset()


def test_load_tags_rejects_unknown_dimension(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[hanon_001]
dimensions = ["timing", "tempo"]
""",
    )
    with pytest.raises(ValueError, match="unknown dimension"):
        load_tags(toml, known_primitive_ids={"hanon_001"})


def test_load_tags_rejects_unknown_primitive(tmp_path: Path):
    toml = _write_toml(
        tmp_path,
        """
[ghost_999]
dimensions = ["timing"]
""",
    )
    with pytest.raises(ValueError, match="unknown primitive_id"):
        load_tags(toml, known_primitive_ids={"hanon_001"})
