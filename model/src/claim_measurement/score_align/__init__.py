"""FRONT 7b (#101): offline score alignment for cached claim bundles.

Annotates each bundle note with ``score_onset`` (global-affine-detrended score
prediction in performance time) + ``bar_number``, the fields the
OnsetDeviationMeasurer (apps/evals claim_taxonomy) consumes. Pure post-processing
over the cached bundles: parangonar re-match on the stored AMT notes, no AMT server.
"""
