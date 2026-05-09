"""Build the Stage 0 capability dossier from probe result files."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from teacher_model.stage0.tier_classifier import classify_tier

_ERROR_RATE_GATE = 0.05  # refuse to emit if any pipeline exceeds 5% errors

# Sonnet baseline lookup -- field name in the aggregate file for outcome means.
_DIM_KEY = "mean_outcome"


class DossierEmissionRefused(RuntimeError):
    """Raised when error rates exceed _ERROR_RATE_GATE."""


@dataclass
class CapabilityRow:
    name: str
    primary_signal: str
    primary_value: float
    primary_ci: tuple[float, float] | None
    corroborating_signal: str | None
    corroborating_value: float | None
    tier: str
    anchor_type: str  # "relative" | "absolute"
    sonnet_baseline: float | None
    delta_vs_sonnet: float | None
    inconsistency_flag: bool
    notes: str = ""


@dataclass
class Dossier:
    meta: dict
    capabilities: list[CapabilityRow]
    continuation_degeneracy_rate: float | None = None
    continuation_by_category: dict[str, int] = field(default_factory=dict)
    over_call_by_category: dict[str, float] = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _synthesis_outcome_means(rows: list[dict]) -> dict[str, list[int]]:
    by_dim: dict[str, list[int]] = {}
    for row in rows:
        if row.get("error"):
            continue
        for d in row.get("judge_dimensions", []):
            outcome = d.get("outcome")
            if isinstance(outcome, (int, float)):
                by_dim.setdefault(d["criterion"], []).append(int(outcome))
    return by_dim


def _bootstrap_ci(values: list[float], n_resamples: int = 1000, seed: int = 1234) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    resamples = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resamples.append(sum(sample) / n)
    resamples.sort()
    lo = resamples[int(0.025 * n_resamples)]
    hi = resamples[int(0.975 * n_resamples) - 1]
    return (lo, hi)


def _wilson_ci_proportion(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    radius = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def _baseline_lookup(baseline: dict, dim: str) -> float:
    for d in baseline.get("dimensions", []):
        if d.get("name") == dim:
            return float(d[_DIM_KEY])
    raise KeyError(f"no baseline entry for {dim!r}")


def _check_error_rate(rows: list[dict], pipeline: str) -> None:
    if not rows:
        return
    errors = sum(1 for r in rows if r.get("error"))
    rate = errors / len(rows)
    if rate > _ERROR_RATE_GATE:
        raise DossierEmissionRefused(
            f"{pipeline}: error rate {rate:.1%} exceeds gate {_ERROR_RATE_GATE:.0%} "
            f"({errors}/{len(rows)})"
        )


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _avg_dim(by_dim: dict[str, list[int]], dims: list[str]) -> tuple[float, list[float]]:
    """Average a per-row mean across multiple dim names; return (overall mean, per-row averages)."""
    if not by_dim or not dims:
        return float("nan"), []
    series_lengths = [len(by_dim.get(d, [])) for d in dims]
    n = min(series_lengths) if series_lengths else 0
    if n == 0:
        return float("nan"), []
    per_row: list[float] = []
    for i in range(n):
        per_row.append(sum(by_dim[d][i] for d in dims) / len(dims))
    return _mean(per_row), per_row


def _build_judgment_row(by_dim: dict[str, list[int]], baseline: dict) -> CapabilityRow:
    ascf_mean = _mean(by_dim.get("Audible-Specific Corrective Feedback", []))
    sgd_mean = _mean(by_dim.get("Scaffolded Guided Discovery", []))
    point = (ascf_mean + sgd_mean) / 2
    ascf_base = _baseline_lookup(baseline, "Audible-Specific Corrective Feedback")
    sgd_base = _baseline_lookup(baseline, "Scaffolded Guided Discovery")
    deltas = [ascf_base - ascf_mean, sgd_base - sgd_mean]
    worst = max(deltas)
    baseline_avg = (ascf_base + sgd_base) / 2
    primary_ci = _bootstrap_ci(_avg_dim(by_dim, ["Audible-Specific Corrective Feedback", "Scaffolded Guided Discovery"])[1])
    tier = classify_tier(value=baseline_avg - worst, baseline=baseline_avg, mode="relative", ci=primary_ci)
    return CapabilityRow(
        name="Judgment",
        primary_signal="avg(ASCF, SGD) outcome",
        primary_value=point,
        primary_ci=primary_ci,
        corroborating_signal=None,
        corroborating_value=None,
        tier=tier,
        anchor_type="relative",
        sonnet_baseline=baseline_avg,
        delta_vs_sonnet=point - baseline_avg,
        inconsistency_flag=False,
    )


def _build_taste_row(by_dim: dict[str, list[int]]) -> CapabilityRow:
    vals = by_dim.get("Taste Defensibility", [])
    point = _mean(vals)
    ci = _bootstrap_ci([float(v) for v in vals])
    return CapabilityRow(
        name="Taste",
        primary_signal="Taste Defensibility (NEW)",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal=None,
        corroborating_value=None,
        tier=classify_tier(value=point, baseline=None, mode="absolute", ci=ci),
        anchor_type="absolute",
        sonnet_baseline=None,
        delta_vs_sonnet=None,
        inconsistency_flag=False,
        notes="no baseline anchor",
    )


def _build_integration_row(by_dim: dict[str, list[int]], baseline: dict) -> CapabilityRow:
    cap = by_dim.get("Concrete Artifact Provision", [])
    cap_mean = _mean(cap)
    cap_ci = _bootstrap_ci([float(v) for v in cap])
    cap_base = _baseline_lookup(baseline, "Concrete Artifact Provision")

    base_dims = [
        "Audible-Specific Corrective Feedback",
        "Concrete Artifact Provision",
        "Specific Positive Praise",
        "Autonomy-Supporting Motivation",
        "Scaffolded Guided Discovery",
        "Style-Consistent Musical Language",
        "Appropriate Tone & Language",
    ]
    composite_mean, _ = _avg_dim(by_dim, base_dims)
    composite_baseline = baseline.get("composite_mean", float("nan"))

    primary_tier = classify_tier(value=cap_mean, baseline=cap_base, mode="relative", ci=cap_ci)
    composite_tier = classify_tier(value=composite_mean, baseline=composite_baseline, mode="relative")
    inconsistent = primary_tier != composite_tier and not _adjacent_tier(primary_tier, composite_tier)
    return CapabilityRow(
        name="Integration",
        primary_signal="CAP outcome",
        primary_value=cap_mean,
        primary_ci=cap_ci,
        corroborating_signal="composite mean (7 base dims)",
        corroborating_value=composite_mean,
        tier=primary_tier,
        anchor_type="relative",
        sonnet_baseline=cap_base,
        delta_vs_sonnet=cap_mean - cap_base,
        inconsistency_flag=inconsistent,
    )


def _build_voice_row(by_dim: dict[str, list[int]], baseline: dict) -> CapabilityRow:
    dims = ["Specific Positive Praise", "Appropriate Tone & Language", "Autonomy-Supporting Motivation"]
    point, series = _avg_dim(by_dim, dims)
    bases = [_baseline_lookup(baseline, d) for d in dims]
    base_mean = sum(bases) / len(bases)
    ci = _bootstrap_ci(series) if series else None
    return CapabilityRow(
        name="Voice",
        primary_signal="avg(SPP, ATL, ASM) outcome",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal=None,
        corroborating_value=None,
        tier=classify_tier(value=point, baseline=base_mean, mode="relative", ci=ci),
        anchor_type="relative",
        sonnet_baseline=base_mean,
        delta_vs_sonnet=point - base_mean,
        inconsistency_flag=False,
    )


def _build_vocabulary_row(by_dim: dict[str, list[int]], baseline: dict, mcq: dict) -> CapabilityRow:
    scml = by_dim.get("Style-Consistent Musical Language", [])
    point = _mean(scml)
    ci = _bootstrap_ci([float(v) for v in scml])
    base_v = _baseline_lookup(baseline, "Style-Consistent Musical Language")
    primary_tier = classify_tier(value=point, baseline=base_v, mode="relative", ci=ci)
    concepts = mcq.get("by_topic", {}).get("concepts", {})
    concepts_acc = float(concepts.get("accuracy", 0.0)) if concepts else 0.0
    inconsistent = (primary_tier == "at_ceiling" and concepts_acc < 0.6)
    return CapabilityRow(
        name="Vocabulary",
        primary_signal="SCML outcome",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal="MCQ concepts accuracy",
        corroborating_value=concepts_acc,
        tier=primary_tier,
        anchor_type="relative",
        sonnet_baseline=base_v,
        delta_vs_sonnet=point - base_v,
        inconsistency_flag=inconsistent,
    )


def _build_tool_calling_row(tool_rows: list[dict]) -> CapabilityRow:
    valid = [r for r in tool_rows if not r.get("error")]
    if not valid:
        point = 0.0
        ci = (0.0, 0.0)
    else:
        correct = sum(1 for r in valid if r.get("discipline_correct"))
        point = correct / len(valid)
        ci = _wilson_ci_proportion(correct, len(valid))
    # absolute thresholds use 0-3 scale; rescale percent so 80% -> 2.4, 50% -> 1.5, <50% -> <1.5
    abs_value = point * 3.0
    abs_ci = (ci[0] * 3.0, ci[1] * 3.0)
    return CapabilityRow(
        name="Tool-calling",
        primary_signal="discipline accuracy",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal="format-conditional schema validity (reported separately)",
        corroborating_value=None,
        tier=classify_tier(value=abs_value, baseline=None, mode="absolute", ci=abs_ci),
        anchor_type="absolute",
        sonnet_baseline=None,
        delta_vs_sonnet=None,
        inconsistency_flag=False,
        notes="no baseline anchor",
    )


def _build_adaptation_row(by_dim: dict[str, list[int]]) -> CapabilityRow:
    vals = by_dim.get("Adaptation Specificity", [])
    point = _mean(vals)
    ci = _bootstrap_ci([float(v) for v in vals])
    return CapabilityRow(
        name="Adaptation",
        primary_signal="Adaptation Specificity (NEW)",
        primary_value=point,
        primary_ci=ci,
        corroborating_signal=None,
        corroborating_value=None,
        tier=classify_tier(value=point, baseline=None, mode="absolute", ci=ci),
        anchor_type="absolute",
        sonnet_baseline=None,
        delta_vs_sonnet=None,
        inconsistency_flag=False,
        notes="no baseline anchor",
    )


def _adjacent_tier(a: str, b: str) -> bool:
    order = ["absent", "mid_tier", "at_ceiling"]
    base_a = a.split("_with_")[0]
    base_b = b.split("_with_")[0]
    if base_a not in order or base_b not in order:
        return True
    return abs(order.index(base_a) - order.index(base_b)) <= 1


def _render_markdown(dossier: Dossier) -> str:
    lines = ["## Capability dossier", ""]
    meta = dossier.meta
    lines.append(f"- model: `{meta.get('model_id', 'unknown')}`")
    if "routed_providers" in meta:
        lines.append(f"- routed_providers: {meta['routed_providers']}")
    lines.append("")
    lines.append("| Capability | Tier | Primary signal | Value | vs Sonnet | CI | Note |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in dossier.capabilities:
        ci_str = f"[{c.primary_ci[0]:.2f},{c.primary_ci[1]:.2f}]" if c.primary_ci else "n/a"
        delta_str = f"{c.delta_vs_sonnet:+.2f}" if c.delta_vs_sonnet is not None else "n/a"
        notes = c.notes
        if c.inconsistency_flag:
            notes = (notes + "; inconsistent primary/corroborator").strip("; ")
        lines.append(
            f"| {c.name} | {c.tier} | {c.primary_signal} | {c.primary_value:.2f} | {delta_str} | {ci_str} | {notes} |"
        )
    if dossier.continuation_degeneracy_rate is not None:
        lines.append("")
        lines.append(
            f"**Continuation degeneracy rate:** {dossier.continuation_degeneracy_rate:.1%}; "
            f"by category: {dossier.continuation_by_category}"
        )
    if dossier.over_call_by_category:
        lines.append("")
        lines.append(f"**Over-call rates by negative category:** {dossier.over_call_by_category}")
    return "\n".join(lines) + "\n"


def build_dossier(
    synthesis_jsonl: Path,
    tool_jsonl: Path,
    mcq_json: Path,
    baseline_aggregate_json: Path,
    out_dir: Path,
    continuation_jsonl: Path | None = None,
) -> Dossier:
    synth_rows = _read_jsonl(synthesis_jsonl)
    tool_rows = _read_jsonl(tool_jsonl)
    _check_error_rate(synth_rows, "synthesis")
    _check_error_rate(tool_rows, "tool")

    mcq = json.loads(mcq_json.read_text())
    baseline = json.loads(baseline_aggregate_json.read_text())

    by_dim = _synthesis_outcome_means(synth_rows)

    capabilities = [
        _build_judgment_row(by_dim, baseline),
        _build_taste_row(by_dim),
        _build_integration_row(by_dim, baseline),
        _build_voice_row(by_dim, baseline),
        _build_vocabulary_row(by_dim, baseline, mcq),
        _build_tool_calling_row(tool_rows),
        _build_adaptation_row(by_dim),
    ]

    routed = sorted({r.get("routed_provider") for r in synth_rows if r.get("routed_provider")})
    meta = {
        "model_id": next((r.get("model_id") for r in synth_rows if r.get("model_id")), "unknown"),
        "n_synthesis": len(synth_rows),
        "n_tool": len(tool_rows),
        "mcq_total": mcq.get("total"),
        "routed_providers": routed,
    }

    # Over-call by category (negatives only)
    neg_rows = [r for r in tool_rows if r.get("expected_call") is False]
    by_cat: dict[str, list[bool]] = {}
    for r in neg_rows:
        c = r.get("category") or "unknown"
        by_cat.setdefault(c, []).append(bool(r.get("called")))
    over_call = {k: sum(v) / len(v) for k, v in by_cat.items() if v}

    cont_rate: float | None = None
    cont_cat: dict[str, int] = {}
    if continuation_jsonl is not None and continuation_jsonl.exists():
        cont_rows = _read_jsonl(continuation_jsonl)
        if cont_rows:
            degen = sum(1 for r in cont_rows if r.get("is_degenerate"))
            cont_rate = degen / len(cont_rows)
            for r in cont_rows:
                cat = r.get("category", "unknown")
                cont_cat[cat] = cont_cat.get(cat, 0) + 1

    dossier = Dossier(
        meta=meta,
        capabilities=capabilities,
        continuation_degeneracy_rate=cont_rate,
        continuation_by_category=cont_cat,
        over_call_by_category=over_call,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capability_dossier.json").write_text(
        json.dumps(
            {
                "meta": dossier.meta,
                "capabilities": [asdict(c) for c in dossier.capabilities],
                "continuation_degeneracy_rate": dossier.continuation_degeneracy_rate,
                "continuation_by_category": dossier.continuation_by_category,
                "over_call_by_category": dossier.over_call_by_category,
            },
            indent=2,
        )
    )
    (out_dir / "capability_dossier.md").write_text(_render_markdown(dossier))
    return dossier
