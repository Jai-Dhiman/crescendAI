"""Front-4 (#101) tolerance (tau) calibration against human perceptual labels.

Calibrates each dimension's tau -- the dead-band half-width the frozen router uses to
decide |d| > tau => anomaly present -- so SUPPORTED matches HUMAN perception of an
anomaly, then flips ``locked: true`` in the taxonomy. No LLM in any label.

  - ``tau_pedaling_render`` : render natural (uncorrupted) PercePiano clips -> aria-amt
    sustain on-fraction, paired with the composite pedaling label (PEP-723, uv run --script).
  - ``tau_calibrate``       : Youden-J-optimal tau on signed d vs composite-rating tails,
    with bootstrap CI and per-direction tau (the flip+ < flip- asymmetry diagnosis).

Method (signed-anomaly detection): a clip is a + anomaly if its composite rating > Q_hi,
- anomaly if < Q_lo, else normal. The verifier fires +1 if d>tau, -1 if d<-tau, else 0.
A detection counts only if it fires with the RIGHT sign on a tail clip; any firing on a
normal clip is a false alarm. tau* = argmax (TPR - FPR). Paths anchored to ``parents[4]``.
"""
