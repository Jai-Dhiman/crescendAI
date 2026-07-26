"""Guard: no measurer render script depends on the deleted aria EndpointHandler.
Run: cd model && uv run --with pytest pytest src/claim_measurement/test_no_aria_amt_handler.py"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SWAPPED = [
    "model/src/claim_measurement/ga_validation/amt_dynamics_ga_render.py",
    "model/src/claim_measurement/ga_validation/amt_dynamics_gb_gate.py",
    "model/src/claim_measurement/ga_validation/amt_pedaling_ga_render.py",
    "model/src/claim_measurement/gc_error_bars/gc_dynamics_render.py",
    "model/src/claim_measurement/tau_calibration/tau_pedaling_render.py",
    "model/src/claim_measurement/amt_fidelity/onset_duration_render.py",
    "model/src/claim_measurement/dynamics_supply/render_percepiano_bundles.py",
    "apps/inference/extract_amt_midi.py",
]


def test_no_endpointhandler_import_remains():
    offenders = []
    for rel in SWAPPED:
        text = (REPO / rel).read_text()
        if "EndpointHandler" in text or "handler._transcribe" in text:
            offenders.append(rel)
    assert offenders == [], f"still on aria handler: {offenders}"


def test_all_use_transkun_cli():
    missing = [rel for rel in SWAPPED
               if "transkun_cli" not in (REPO / rel).read_text()]
    assert missing == [], f"not swapped to transkun_cli: {missing}"
