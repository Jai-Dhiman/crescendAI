"""GATE 1 localization-robustness measurement against the SHIPPED LocationResolver.

For each bar present in the clean score's measure_table, resolve its start time
in the clean bundle (M_clean) and the corrupted bundle (M_corrupt), then compare
M_corrupt against the construction-known prediction W(M_clean). The signed
deviation delta = M_corrupt - W(M_clean) is the localization error injected by the
corruption -- measured with no hand-annotated ground truth, because W is known.

This module imports the production resolver so the number reflects exactly what
the verifier would localize; it never invokes an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from claim_taxonomy.verifier.location_resolver import LocationResolver
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


@dataclass(frozen=True)
class BarDelta:
    bar: int
    clean_sec: float | None
    corrupt_sec: float | None
    predicted_sec: float | None
    delta_sec: float | None
    bar_dur_sec: float | None
    resolvable: bool


def warp_time(warp_map: dict, t: float) -> float:
    """Map a clean-time second to corrupt-time under the piecewise-linear warp map."""
    return float(np.interp(t, warp_map["clean_sec"], warp_map["corrupt_sec"]))


def _resolve_bar_start(resolver: LocationResolver, bar: int) -> float:
    """Resolved audio-time start of a single bar (raises UnverifiableError)."""
    region = resolver.resolve({"bar_start": bar, "bar_end": bar})
    return region.audio_start_sec


def _bar_durations(measure_table: list[dict]) -> dict[int, float]:
    rows = sorted(measure_table, key=lambda r: int(r["bar_number"]))
    durs: dict[int, float] = {}
    for i, row in enumerate(rows):
        bar = int(row["bar_number"])
        if i + 1 < len(rows):
            durs[bar] = float(rows[i + 1]["start_sec"]) - float(row["start_sec"])
        elif i > 0:
            durs[bar] = float(row["start_sec"]) - float(rows[i - 1]["start_sec"])
        else:
            durs[bar] = float("nan")
    return durs


def bar_localization_deltas(
    clean_bundle: dict,
    corrupt_bundle: dict,
    warp_map: dict,
    engine: SubstrateErrorEngine | None = None,
) -> list[BarDelta]:
    """Per-bar localization deviation of corrupt vs warp-predicted clean."""
    if engine is None:
        engine = SubstrateErrorEngine(seed=42)
    clean_resolver = LocationResolver(clean_bundle, engine)
    corrupt_resolver = LocationResolver(corrupt_bundle, engine)
    durs = _bar_durations(clean_bundle["measure_table"])
    bars = sorted(int(r["bar_number"]) for r in clean_bundle["measure_table"])

    out: list[BarDelta] = []
    for bar in bars:
        bar_dur = durs.get(bar)
        try:
            clean_sec = _resolve_bar_start(clean_resolver, bar)
        except UnverifiableError:
            clean_sec = None
        try:
            corrupt_sec = _resolve_bar_start(corrupt_resolver, bar)
        except UnverifiableError:
            corrupt_sec = None

        if clean_sec is None or corrupt_sec is None:
            out.append(BarDelta(bar, clean_sec, corrupt_sec, None, None, bar_dur, False))
            continue

        predicted = warp_time(warp_map, clean_sec)
        out.append(
            BarDelta(
                bar=bar,
                clean_sec=clean_sec,
                corrupt_sec=corrupt_sec,
                predicted_sec=predicted,
                delta_sec=corrupt_sec - predicted,
                bar_dur_sec=bar_dur,
                resolvable=True,
            )
        )
    return out


def accuracy_at_tolerances(
    deltas: list[BarDelta], tolerances_sec: list[float]
) -> dict:
    """Aggregate per-bar deltas into resolvable-rate and within-tolerance accuracy.

    within_over_total counts unresolvable bars as failures (strict, the localization
    yield the verifier actually achieves); within_over_resolvable conditions on the
    bars the resolver attempted. Both are reported -- an unresolvable bar yields a
    SAFE UNVERIFIABLE verdict, not a wrong answer, so the two framings differ.
    """
    n_total = len(deltas)
    resolvable = [d for d in deltas if d.resolvable]
    n_res = len(resolvable)
    abs_deltas = np.array([abs(d.delta_sec) for d in resolvable]) if resolvable else np.array([])

    tol_out: dict[str, dict] = {}
    for tol in tolerances_sec:
        n_within = int(np.sum(abs_deltas <= tol)) if abs_deltas.size else 0
        tol_out[str(tol)] = {
            "within_over_total": (n_within / n_total) if n_total else 0.0,
            "within_over_resolvable": (n_within / n_res) if n_res else 0.0,
            "n_within": n_within,
        }

    return {
        "n_total": n_total,
        "n_resolvable": n_res,
        "resolvable_rate": (n_res / n_total) if n_total else 0.0,
        "abs_delta_median": float(np.median(abs_deltas)) if abs_deltas.size else None,
        "abs_delta_p90": float(np.percentile(abs_deltas, 90)) if abs_deltas.size else None,
        "abs_delta_max": float(np.max(abs_deltas)) if abs_deltas.size else None,
        "tolerances": tol_out,
    }
