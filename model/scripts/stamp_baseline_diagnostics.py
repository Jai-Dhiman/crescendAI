"""Extract Chunk A diagnostic numbers from a completed sweep results JSON and
stamp them into the P0 section of all four Wave 1 plan files.

Usage:
    cd model/
    uv run python scripts/stamp_baseline_diagnostics.py [--results-path PATH] [--config NAME]

If --config is omitted, numbers are taken from the config with the highest
pairwise_mean (the sweep winner).  If results_path is omitted it defaults to
data/results/a1_max_sweep_results.json.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "data" / "results" / "a1_max_sweep_results.json"
PLAN_DIR = REPO_ROOT / "docs" / "plans"

PLAN_FILES = [
    "2026-04-20-percepiano-anchor-emphasis.md",
    "2026-04-20-heteroscedastic-heads.md",
    "2026-04-20-semi-sup-con-loss.md",
    "2026-04-20-practice-augmentation.md",
]

DIMENSIONS = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]

BASELINE_ANCHOR = "<!-- BASELINE_DIAGNOSTICS -->"


def _mean_matrix(matrices: list) -> np.ndarray:
    """Element-wise mean of a list of 6x6 matrices (each may be a nested list or None)."""
    valid = [np.array(m) for m in matrices if m is not None]
    if not valid:
        return np.full((6, 6), float("nan"))
    return np.mean(np.stack(valid, axis=0), axis=0)


def _format_matrix_md(mat: np.ndarray, dims: list[str]) -> str:
    col_w = 7
    header = " " * 14 + "  ".join(f"{d[:col_w]:>{col_w}}" for d in dims)
    rows = [header]
    for i, row_dim in enumerate(dims):
        row = f"{row_dim:<14}" + "  ".join(f"{mat[i, j]:>{col_w}.3f}" for j in range(len(dims)))
        rows.append(row)
    return "\n".join(rows)


def _format_skill_disc(skill_data: list) -> str:
    """Summarise skill_discrimination_per_fold list into a readable string."""
    if not skill_data:
        return "`skipped` — no T5 tier labels available in PercePiano CV"
    reasons = set()
    for fold_entry in skill_data:
        if fold_entry is None:
            reasons.add("null fold entry")
            continue
        if "skipped" in fold_entry:
            reasons.add(fold_entry["skipped"])
        elif "per_tier_pair" in fold_entry:
            pairs = fold_entry["per_tier_pair"]
            if not pairs:
                reasons.add("insufficient_tier_diversity")
    if reasons:
        reason_str = ", ".join(sorted(reasons))
        return (
            f"`skipped` — {reason_str}\n"
            "  Requires `data/evals/ood_practice/labels.json` populated with T5 tier labels."
        )
    return "(available — see results JSON)"


def _build_baseline_block(config_name: str, config_data: dict) -> str:
    collapse_mean = config_data.get("dimension_collapse_mean")
    collapse_str = f"{collapse_mean:.4f}" if collapse_mean is not None else "n/a"

    pw_mean = config_data.get("pairwise_mean", float("nan"))
    r2_mean = config_data.get("r2_mean", float("nan"))

    per_fold_corr = config_data.get("per_dimension_correlation_per_fold", [])
    corr_mean = _mean_matrix(per_fold_corr)

    skill_per_fold = config_data.get("skill_discrimination_per_fold", [])
    skill_str = _format_skill_disc(skill_per_fold)

    collapse_per_fold = config_data.get("dimension_collapse_per_fold", [])
    collapse_fold_str = ", ".join(
        f"{v:.4f}" if v is not None else "n/a" for v in collapse_per_fold
    )

    matrix_md = _format_matrix_md(corr_mean, DIMENSIONS)

    block = textwrap.dedent(f"""\
        {BASELINE_ANCHOR}
        #### A1-Max baseline diagnostics (config: `{config_name}`, 4-fold CV)

        | Metric | Value |
        |--------|-------|
        | `dimension_collapse_mean` | **{collapse_str}** |
        | `dimension_collapse_per_fold` | {collapse_fold_str} |
        | Pairwise accuracy (4-fold mean) | {pw_mean:.4f} |
        | R² (4-fold mean) | {r2_mean:.4f} |
        | Skill discrimination Cohen's d | {skill_str} |

        **Per-dimension prediction correlation matrix (element-wise mean across folds):**

        ```
        {matrix_md}
        ```

        > Numbers captured from `data/results/a1_max_sweep_results.json`.
        > Re-run `model/scripts/stamp_baseline_diagnostics.py` to refresh after the sweep completes.
        """)
    return block


def _stamp_plan(plan_path: Path, block: str) -> None:
    text = plan_path.read_text()

    # Remove previous stamped block if present.
    if BASELINE_ANCHOR in text:
        start = text.index(BASELINE_ANCHOR)
        # Find next top-level heading after the anchor.
        rest = text[start:]
        next_heading = rest.find("\n## ", 1)
        if next_heading == -1:
            next_heading = rest.find("\n### ", 1)
        if next_heading == -1:
            next_heading = len(rest)
        text = text[:start] + text[start + next_heading:].lstrip("\n")

    # Insert after the P0 heading.
    p0_markers = ["### P0 —", "### P0—", "## P0 —", "## P0—"]
    insert_at = -1
    for marker in p0_markers:
        if marker in text:
            idx = text.index(marker)
            # Skip past the P0 heading line.
            end_of_line = text.index("\n", idx)
            insert_at = end_of_line + 1
            break

    if insert_at == -1:
        # Fallback: append at end.
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    else:
        text = text[:insert_at] + "\n" + block + "\n" + text[insert_at:]

    plan_path.write_text(text)
    print(f"  Stamped: {plan_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config name to use for baseline (default: highest pairwise_mean)"
    )
    args = parser.parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {args.results_path}\n"
            "Run the A1-Max sweep first:\n"
            "  cd model && uv run python -m model_improvement.a1_max_sweep"
        )

    with open(args.results_path) as f:
        results = json.load(f)

    if args.config:
        if args.config not in results:
            raise KeyError(f"Config '{args.config}' not found in results. Available: {list(results)}")
        config_name = args.config
    else:
        config_name = max(results, key=lambda k: results[k].get("pairwise_mean", 0.0))

    config_data = results[config_name]
    print(f"Using config: {config_name}")
    print(f"  pairwise_mean: {config_data.get('pairwise_mean', 'n/a'):.4f}")
    print(f"  dimension_collapse_mean: {config_data.get('dimension_collapse_mean', 'n/a')}")

    block = _build_baseline_block(config_name, config_data)

    for plan_file in PLAN_FILES:
        plan_path = PLAN_DIR / plan_file
        if not plan_path.exists():
            print(f"  Skipping (not found): {plan_file}")
            continue
        _stamp_plan(plan_path, block)

    print(f"\nDone. Baseline block stamped into {len(PLAN_FILES)} plan files.")


if __name__ == "__main__":
    main()
