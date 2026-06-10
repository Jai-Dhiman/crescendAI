"""Editorial technique-tag layer for the exercise corpus.

Each exercise primitive is tagged with the teacher DIMENSIONS it can address
plus free-vocabulary technique labels. Retrieval (match.match_by_dimension)
filters candidates by these dimensions. Tags are hand-authored editorial data,
so they live in a version-controlled technique_tags.toml -- never baked into the
machine-regenerated SQLite catalog, which would silently drop them on rebuild.

load_tags validates the table against the catalog so authoring drift (a typo in
a dimension, or a tag for a primitive that no longer exists) fails loudly.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

# The canonical 6 teacher dimensions (mirrors
# apps/api/src/harness/artifacts/diagnosis.ts DIMENSIONS).
DIMENSIONS = (
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
)


@dataclass(frozen=True)
class TagSet:
    """The dimensions an exercise can address plus free technique labels."""

    dimensions: frozenset[str]
    techniques: frozenset[str]
    key: str


def load_tags(path: Path, known_primitive_ids: set[str]) -> dict[str, TagSet]:
    """Read and validate technique_tags.toml into a {primitive_id: TagSet} map.

    Args:
        path: path to the technique_tags.toml file.
        known_primitive_ids: the primitive_ids present in the catalog; every
            tagged primitive must be one of these.

    Returns:
        Dict mapping primitive_id to its TagSet.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if a tag references a primitive_id absent from the catalog,
            or declares a dimension outside the canonical 6.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"technique tags file not found: {path}")
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    tags: dict[str, TagSet] = {}
    for primitive_id, entry in raw.items():
        if primitive_id not in known_primitive_ids:
            raise ValueError(
                f"tag references unknown primitive_id {primitive_id!r} "
                f"(not in catalog)"
            )
        dims = tuple(entry.get("dimensions", ()))
        for d in dims:
            if d not in DIMENSIONS:
                raise ValueError(
                    f"unknown dimension {d!r} for {primitive_id!r}; "
                    f"valid dimensions are {DIMENSIONS}"
                )
        techs = tuple(entry.get("techniques", ()))
        raw_key = entry.get("key")
        if raw_key is None:
            raise ValueError(
                f"missing required 'key' field for {primitive_id!r} in {path}"
            )
        tags[primitive_id] = TagSet(
            dimensions=frozenset(dims), techniques=frozenset(techs), key=raw_key
        )
    return tags
