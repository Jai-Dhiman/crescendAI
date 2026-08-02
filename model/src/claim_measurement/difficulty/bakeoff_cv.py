"""Composer-disjoint CV + Kendall tau-c, ported from the unmerged
issue-104-mirex-difficulty branch's phase5b_aria_probe.py (commit 7976b5e6).

Ported, not imported cross-branch: worktrees are separate checkouts and this
file does not exist on issue-138-encoder-finetune or on main. The RidgeCV
pipeline, alpha grid, and tau-c convention are unchanged from the original;
the private `_folds` closure inside the original `_oof_tau` is promoted to
the public `composer_disjoint_folds` (added in Task 3) so it is independently
testable per the #138 Phase 0 design's TDD targets.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def tau_c(x, y) -> float | None:
    """Kendall tau-c, nan-safe. None (never 0.0) when the input cannot
    support a rank correlation -- fewer than 3 points, or either side
    constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)
