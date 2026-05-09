"""Map composer string to musical era for sample-stratification quotas.

Only the 4 composers present in apps/evals/results/baseline_v1.jsonl are
recognized. Unknown composers map to "Other" and do not contribute to era
quotas in select_sample.
"""
from __future__ import annotations

_COMPOSER_TO_ERA: dict[str, str] = {
    "Bach": "Baroque",
    "Beethoven": "Classical",
    "Chopin": "Romantic",
    "Debussy": "Impressionist",
}


def composer_to_era(composer: str) -> str:
    return _COMPOSER_TO_ERA.get(composer, "Other")
