"""Tests for build_context.py's model-dir layout validation (#175).

build_context.py runs outside the isolated model script env (plain stdlib), so
it deliberately does not import claim_measurement.difficulty.score_wav's
discover_members -- that would drag in peft/torch. It duplicates the two-layout
rule instead; these tests keep the duplicate honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_context import CODE_FILES, DOCKER_FILES, build_context


def _make_stub_tree(tmp_path: Path) -> dict:
    """Build a minimal fake tree satisfying every build_context input except
    the model_dir layout, which each test sets up separately."""
    difficulty_dir = tmp_path / "difficulty"
    difficulty_dir.mkdir()
    for name in CODE_FILES:
        (difficulty_dir / name).write_text("# stub\n")

    amt_dir = tmp_path / "amt"
    amt_dir.mkdir()
    (amt_dir / "transkun_cli.py").write_text("# stub\n")

    moonbeam_dir = tmp_path / "moonbeam"
    (moonbeam_dir / "repo" / "src" / "llama_recipes" / "configs").mkdir(
        parents=True)
    (moonbeam_dir / "moonbeam_839M.pt").write_bytes(b"stub")
    (moonbeam_dir / "repo" / "src" / "llama_recipes" / "configs"
     / "model_config.json").write_text("{}")

    here = tmp_path / "here"
    here.mkdir()
    for name in DOCKER_FILES:
        (here / name).write_text("# stub\n")

    return {
        "difficulty_dir": difficulty_dir,
        "amt_dir": amt_dir,
        "moonbeam_dir": moonbeam_dir,
        "here": here,
    }


def test_build_context_accepts_a_members_layout_model_dir(tmp_path):
    stubs = _make_stub_tree(tmp_path)
    model_dir = tmp_path / "model_dir"
    member_dir = model_dir / "members" / "s2026"
    (member_dir / "adapter").mkdir(parents=True)
    (member_dir / "adapter" / "adapter_config.json").write_text("{}")
    (member_dir / "ridge_head.npz").write_bytes(b"stub")

    out_dir = tmp_path / "out"
    report = build_context(out_dir=out_dir, model_dir=model_dir,
                           **stubs)

    assert report["out_dir"] == str(out_dir)
    assert (out_dir / "model" / "members" / "s2026" / "adapter"
            / "adapter_config.json").exists()
    assert (out_dir / "model" / "members" / "s2026" / "ridge_head.npz").exists()


def test_build_context_still_accepts_a_legacy_model_dir(tmp_path):
    stubs = _make_stub_tree(tmp_path)
    model_dir = tmp_path / "model_dir"
    (model_dir / "adapter").mkdir(parents=True)
    (model_dir / "adapter" / "adapter_config.json").write_text("{}")
    (model_dir / "ridge_head.npz").write_bytes(b"stub")

    out_dir = tmp_path / "out"
    report = build_context(out_dir=out_dir, model_dir=model_dir,
                           **stubs)

    assert report["out_dir"] == str(out_dir)
    assert (out_dir / "model" / "adapter" / "adapter_config.json").exists()
    assert (out_dir / "model" / "ridge_head.npz").exists()
