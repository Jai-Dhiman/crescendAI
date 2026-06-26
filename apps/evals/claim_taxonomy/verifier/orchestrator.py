from __future__ import annotations

from claim_taxonomy.verifier.location_resolver import LocationResolver
from claim_taxonomy.verifier.models import UnverifiableError, VerdictResult
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine
from claim_taxonomy.verdict_dispatch import route_verdict


def _build_registry():
    from claim_taxonomy.verifier.measurers.timing import TimingMeasurer
    from claim_taxonomy.verifier.measurers.pedaling import PedalingMeasurer
    from claim_taxonomy.verifier.measurers.dynamics import DynamicsMeasurer
    return {
        "amt_onsets_region_tempo_fit": TimingMeasurer(),
        "amt_sustain_pedal_events": PedalingMeasurer(),
        "amt_note_velocity_estimator": DynamicsMeasurer(),
    }


_MEASURER_REGISTRY = None


def _get_registry() -> dict:
    global _MEASURER_REGISTRY
    if _MEASURER_REGISTRY is None:
        _MEASURER_REGISTRY = _build_registry()
    return _MEASURER_REGISTRY


def verify(
    claim: dict,
    bundle: dict,
    taxonomy: dict,
    engine: SubstrateErrorEngine | None = None,
) -> VerdictResult:
    """Full verification pipeline for one claim against one bundle.

    Never raises. Returns VerdictResult for all outcomes including UNVERIFIABLE.
    """
    if engine is None:
        engine = SubstrateErrorEngine(seed=42)

    registry = taxonomy["dimensions"]
    dimension_name = claim["dimension"]
    location = claim["location"]
    substrate_versions = bundle.get("substrate_versions", {})

    dim = registry.get(dimension_name)
    if dim is None:
        return VerdictResult(
            verdict="UNVERIFIABLE", reason_code="out_of_scope_dim",
            measured_value=0.0, tau=0.0, error_bar=0.0, event_count=0, units="",
            substrate_versions=substrate_versions, dimension=dimension_name, location=location,
        )

    if dim["status"] == "scoped_out":
        return VerdictResult(
            verdict="UNVERIFIABLE", reason_code="out_of_scope_dim",
            measured_value=0.0, tau=0.0, error_bar=0.0, event_count=0, units="",
            substrate_versions=substrate_versions, dimension=dimension_name, location=location,
        )

    if dim["status"] == "gated_on_measurement":
        return VerdictResult(
            verdict="UNVERIFIABLE", reason_code="gated_dim",
            measured_value=0.0, tau=0.0, error_bar=0.0, event_count=0, units="",
            substrate_versions=substrate_versions, dimension=dimension_name, location=location,
        )

    tau = float(dim["tolerance"]["provisional"])
    units = dim["tolerance"]["unit"]
    measurement_key = dim["measurement"]
    measurer_registry = _get_registry()

    if measurement_key not in measurer_registry:
        return VerdictResult(
            verdict="UNVERIFIABLE", reason_code="substrate_failure",
            measured_value=0.0, tau=tau, error_bar=0.0, event_count=0, units=units,
            substrate_versions=substrate_versions, dimension=dimension_name, location=location,
        )

    min_coverage = float(
        taxonomy.get("localization_granularity", {})
        .get("coverage_gate", {})
        .get("threshold", 0.0)
    )
    try:
        resolver = LocationResolver(bundle, engine, min_coverage=min_coverage)
        region = resolver.resolve(location)
    except UnverifiableError as e:
        return VerdictResult(
            verdict="UNVERIFIABLE", reason_code=e.reason_code,
            measured_value=0.0, tau=tau, error_bar=0.0, event_count=0, units=units,
            substrate_versions=substrate_versions, dimension=dimension_name, location=location,
        )

    measurer = measurer_registry[measurement_key]
    try:
        measurement = measurer.measure(
            location=location, bundle=bundle, region=region, engine=engine,
        )
    except UnverifiableError as e:
        return VerdictResult(
            verdict="UNVERIFIABLE", reason_code=e.reason_code,
            measured_value=0.0, tau=tau, error_bar=0.0, event_count=0, units=units,
            substrate_versions=substrate_versions, dimension=dimension_name, location=location,
        )

    if measurement.substrate_failure:
        return VerdictResult(
            verdict="UNVERIFIABLE", reason_code="substrate_failure",
            measured_value=measurement.d, tau=tau, error_bar=measurement.error_bar,
            event_count=measurement.event_count, units=units,
            substrate_versions=substrate_versions, dimension=dimension_name, location=location,
        )

    populated_claim = dict(claim)
    populated_claim["_measurement"] = {
        "d": measurement.d,
        "tau": tau,
        "error_bar": measurement.error_bar,
        "event_count": measurement.event_count,
        "localizable": True,
        "substrate_failure": False,
    }

    verdict, reason_code = route_verdict(populated_claim, registry)

    return VerdictResult(
        verdict=verdict, reason_code=reason_code,
        measured_value=measurement.d, tau=tau, error_bar=measurement.error_bar,
        event_count=measurement.event_count, units=units,
        substrate_versions=substrate_versions, dimension=dimension_name, location=location,
    )
