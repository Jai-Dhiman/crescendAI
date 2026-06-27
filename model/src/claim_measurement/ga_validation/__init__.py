"""Construction-known G-A / G-B validation harnesses for the claim verifier (#101).

These reproduce the hard-gate evidence cited in
``docs/model/claim-verifier-signed-d-conventions.md`` (GATE 3 UPDATE + Path #1 gates):
render PercePiano MIDI at fixed gain, transcribe with aria-amt, and feed the result to
the REAL verifier measurers + frozen ``route_verdict``. No LLM is in any truth label.

Two stages per dimension:
  - ``*_ga_render``  : construction-known corruption -> fluidsynth -> aria-amt -> JSON
                       (PEP-723 self-contained; run with ``uv run --script``).
  - ``*_ga_metrics`` : load the render JSON, run the real measurer + frozen router,
                       compute performance-flip + polarity-shuffle (run in apps/evals env).
  - ``amt_dynamics_gb_gate`` : end-to-end G-B perceptual-validity correlation.

All paths are anchored to this module (``REPO = parents[4]``), never CWD-relative.
"""
