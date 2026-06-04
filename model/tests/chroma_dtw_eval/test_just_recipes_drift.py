"""Locks the rework's recipe-naming contract. Asserts the four recipes
the rework introduces exist by name in Justfile, by parsing the file
directly (no `just` binary required).
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
JUSTFILE = REPO_ROOT / "Justfile"


@pytest.mark.parametrize("recipe", [
    "chroma-eval-verify",
    "chroma-eval-verify-smoke",
    "chroma-eval-ratchet",
    "chroma-eval-prebuild",
    "amt-regen-pseudo-truth",
])
def test_recipe_present_in_justfile(recipe: str) -> None:
    assert JUSTFILE.exists(), f"Justfile missing: {JUSTFILE}"
    body = JUSTFILE.read_text()
    # Recipes appear as `name:` or `name arg1 arg2:` at start of line.
    found = any(
        line.split(":")[0].split()[0] == recipe
        for line in body.splitlines()
        if line and not line.startswith((" ", "\t", "#"))
    )
    assert found, f"missing recipe {recipe!r} in {JUSTFILE}"
