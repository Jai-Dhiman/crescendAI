"""LLM distillation pilot: rubric, scoring, and calibration."""

from __future__ import annotations

import json
import re

import numpy as np
from scipy.stats import pearsonr


def build_rubric(
    taxonomy: dict[str, dict],
    quote_bank: dict[str, list[dict]],
) -> dict[str, dict]:
    """Build a 1-5 scoring rubric for each taxonomy dimension.

    Uses quote_bank examples to ground the anchor descriptions.

    Returns:
        {dim_name: {description, anchors: {1: str, ..., 5: str}}}.
    """
    rubric = {}
    for dim_name, dim_info in taxonomy.items():
        desc = dim_info.get("description", dim_name)
        quotes = quote_bank.get(dim_name, [])

        # Extract negative and positive examples from quotes
        negative = [q["feedback_summary"] for q in quotes if q.get("severity") in ("critical", "significant")]
        positive = [q["feedback_summary"] for q in quotes if q.get("severity") == "minor" and q.get("feedback_type") == "praise"]

        neg_example = negative[0] if negative else "Poor quality"
        pos_example = positive[0] if positive else "Excellent quality"

        rubric[dim_name] = {
            "description": desc,
            "anchors": {
                1: f"Severely lacking in {dim_name}. E.g.: {neg_example}",
                2: f"Below average {dim_name}; noticeable issues.",
                3: f"Adequate {dim_name}; competent but not distinctive.",
                4: f"Good {dim_name}; shows musical awareness.",
                5: f"Excellent {dim_name}. E.g.: {pos_example}",
            },
        }
    return rubric


def build_scoring_prompt(rubric: dict[str, dict], segment_id: str) -> str:
    """Build the user prompt for LLM scoring of a segment."""
    lines = [
        f"Score this piano performance segment: {segment_id}\n",
        "Rate each dimension on a 1-5 scale:\n",
    ]
    for dim_name, entry in rubric.items():
        lines.append(f"## {dim_name}: {entry['description']}")
        for score, anchor in entry["anchors"].items():
            lines.append(f"  {score} - {anchor}")
        lines.append("")

    lines.append(
        'Respond with a JSON object mapping each dimension to its score (integer 1-5).\n'
        "Example: " + json.dumps({d: 3 for d in rubric})
    )
    return "\n".join(lines)


def parse_scores(response: str, dimensions: list[str]) -> dict[str, int]:
    """Parse LLM scoring response into {dim: score} dict."""
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON in response: {text[:200]}")

    # Find matching brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = json.loads(text[start : i + 1])
                break
    else:
        raise ValueError("Unterminated JSON")

    scores = {}
    for dim in dimensions:
        val = raw.get(dim, 3)
        scores[dim] = max(1, min(5, int(round(val))))
    return scores


def calibration_analysis(
    teacher_scores: np.ndarray,
    composite_labels: np.ndarray,
    dim_name: str,
    r_threshold: float = 0.5,
    bias_threshold: float = 0.5,
) -> dict:
    """Compare teacher LLM scores against PercePiano composite labels.

    Args:
        teacher_scores: [N] array of teacher 1-5 scores (normalized to 0-1).
        composite_labels: [N] array of composite labels (0-1).
        dim_name: Dimension name for reporting.

    Returns:
        Dict with pearson_r, mean_offset, passed_correlation, passed_bias.
    """
    r, p_value = pearsonr(teacher_scores, composite_labels)
    mean_offset = abs(float(np.mean(teacher_scores) - np.mean(composite_labels)))

    return {
        "dimension": dim_name,
        "pearson_r": float(r),
        "p_value": float(p_value),
        "mean_offset": mean_offset,
        "passed_correlation": r > r_threshold,
        "passed_bias": mean_offset < bias_threshold,
    }


def go_no_go(
    per_dim_results: dict[str, dict],
    stop_auc: float,
    spot_check_accuracy: float,
    min_passing_frac: float = 0.60,
    min_stop_auc: float = 0.75,
    min_spot_check: float = 0.50,
) -> dict:
    """Go/no-go decision for scaling distillation to T2/T3/T4.

    Args:
        per_dim_results: {dim: calibration_analysis result}.
        stop_auc: Teacher-label STOP prediction AUC.
        spot_check_accuracy: Fraction of masterclass moments where teacher's
            lowest-scoring dimension matches actual feedback category.
    """
    n_total = len(per_dim_results)
    n_passing = sum(1 for r in per_dim_results.values() if r["passed_correlation"])
    frac_passing = n_passing / n_total if n_total > 0 else 0.0

    go = (
        frac_passing >= min_passing_frac
        and stop_auc >= min_stop_auc
        and spot_check_accuracy >= min_spot_check
    )

    return {
        "go": go,
        "dims_passing": n_passing,
        "dims_total": n_total,
        "frac_passing": frac_passing,
        "stop_auc": stop_auc,
        "spot_check_accuracy": spot_check_accuracy,
    }
