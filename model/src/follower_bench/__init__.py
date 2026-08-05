"""Score-follower matcher core (issue #111, reduced by #133).

This package began as the synthetic clip benchmark. That machinery -- the clip
generator, spliced pathologies, and their trajectory/metric scoring -- was
retired once the real-audio eval passed its PASS bars
(docs/model/realaudio-follower-eval.md). What remains is the matcher itself
(the #115 monotonic DP -> #118 jump-aware DP -> #119 HMM line) plus the
score-note loader, performance-note segmentation, and ASAP alignment reader that
``follower_eval`` imports.
"""
