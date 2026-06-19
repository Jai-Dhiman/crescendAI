# apps/evals/claim_taxonomy/verdict_dispatch.py
"""Verdict routing stub for the claim taxonomy verifier.

Control-flow dispatch only — no real measurement is performed here.
The verifier (issue #65) supplies measurement context via the claim's
`_measurement` dict. This module routes the claim through the 9-step
dispatch chain and returns (verdict, reason_code | None).

Explicit exception handling: unknown dimension or missing fields raise
TypeError immediately (never silently degrade).
"""
from __future__ import annotations


def route_verdict(
    claim: dict,
    registry: dict,
) -> tuple[str, str | None]:
    """Route a structured claim to SUPPORTED | REFUTED | UNVERIFIABLE.

    Args:
        claim: A structured claim dict. Must include:
            - dimension: str
            - polarity: "+" | "-" | "neutral"
            - _measurement: dict with keys:
                - d: float              signed deviation from reference
                - tau: float            tolerance threshold
                - error_bar: float      measurement error bar
                - event_count: int      events found in the region
                - localizable: bool     whether the location was resolved
                - substrate_failure: bool (optional, default False)
        registry: The dimensions dict from claim_taxonomy.json.

    Returns:
        (verdict, reason_code) where verdict in {SUPPORTED, REFUTED, UNVERIFIABLE}
        and reason_code is one of the typed codes or None.

    Raises:
        TypeError: if dimension is not in registry, or _measurement is missing.
    """
    if "dimension" not in claim:
        raise TypeError("claim must include a 'dimension' key.")
    dimension_name = claim["dimension"]
    if dimension_name not in registry:
        raise TypeError(
            f"Unknown dimension '{dimension_name}'. "
            f"Known: {list(registry.keys())}"
        )

    if "_measurement" not in claim:
        raise TypeError(
            "claim must include a '_measurement' dict with measurement context. "
            "The verifier (issue #65) populates this before calling route_verdict."
        )

    dim = registry[dimension_name]
    m = claim["_measurement"]
    polarity = claim["polarity"]

    # Step 1: scoped_out
    if dim["status"] == "scoped_out":
        return ("UNVERIFIABLE", "out_of_scope_dim")

    # Step 2: gated_on_measurement
    if dim["status"] == "gated_on_measurement":
        return ("UNVERIFIABLE", "gated_dim")

    # Step 3: not localizable
    if "localizable" not in m:
        raise TypeError(
            "_measurement must include 'localizable' (bool). "
            "The verifier (issue #65) resolves the location before calling route_verdict."
        )
    if not m["localizable"]:
        return ("UNVERIFIABLE", "unlocalizable")

    # Step 4: substrate failure
    if m.get("substrate_failure", False):
        return ("UNVERIFIABLE", "substrate_failure")

    # Step 5: region too short
    minimum_events = dim.get("minimum_events", 1)
    if m["event_count"] < minimum_events:
        return ("UNVERIFIABLE", "region_too_short")

    d = m["d"]
    tau = m["tau"]
    error_bar = m["error_bar"]

    # Step 6: compute signed deviation d vs reference — performed by the verifier
    # (#65) and supplied via _measurement; this stub consumes d/tau/error_bar directly.

    # Step 7: near threshold
    if abs(abs(d) - tau) <= error_bar:
        return ("UNVERIFIABLE", "near_threshold")

    # Step 8 & 9: polarity confirmation
    if polarity == "+":
        supported = d > 0 and abs(d) > tau
    elif polarity == "-":
        supported = d < 0 and abs(d) > tau
    else:
        # neutral: asserts virtue / absence-of-problem -> supported if no anomaly
        supported = abs(d) <= tau

    if supported:
        return ("SUPPORTED", None)
    return ("REFUTED", None)
