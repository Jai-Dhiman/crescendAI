"""G-C empirical (not assumed) error bars for the claim verifier (#101).

Replaces the placeholder substrate-noise constant in DynamicsMeasurer
(``VELOCITY_QUANT_STEP``, a step/sqrt(12) guess) with the MEASURED re-transcription
churn of the G-B-validated whole_piece dynamics statistic (mean AMT note-velocity), so
the frozen router's near-threshold dead-band (``verdict_dispatch.py``:
``abs(abs(d)-tau) <= error_bar``) is set to >= the measured 1-sigma.

aria-amt decodes greedily (``transcription.py`` argmax; ``model.eval()``), so
re-transcribing an identical WAV is a no-op (verified per clip). The churn is measured
under perceptually neutral recording nuisances -- sub-JND gain jitter + high-SNR
additive white noise -- i.e. nuisance-equivalent re-captures of the SAME performance.

Two stages (mirrors ga_validation / tau_calibration), paths anchored to ``parents[4]``:
  - ``gc_dynamics_render`` : render -> K nuisance re-captures -> aria-amt -> per-variant
                             velocities + matched per-note deltas JSON (PEP-723,
                             ``uv run --script``).
  - ``gc_churn_report``    : reduce to per-clip + pooled 1-sigma, recommend the per-note
                             substrate sigma, and print the placeholder it replaces
                             (run in apps/evals env).
  - ``gc_churn_metrics``   : pure, dependency-free reduction math (unit-tested in
                             ``test_gc_churn_metrics``).
"""
