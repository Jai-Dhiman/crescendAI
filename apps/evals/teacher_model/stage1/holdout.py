import random
from collections import defaultdict

from teacher_model.stage1.briefing_source import Briefing


def split_holdout(
    briefings: list[Briefing],
    frac: float,
    strata: list[str],
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")

    rng = random.Random(seed)

    by_stratum: dict[tuple, list[str]] = defaultdict(list)
    for b in briefings:
        key = tuple(getattr(b, s) for s in strata)
        by_stratum[key].append(b.briefing_id)

    train: list[str] = []
    holdout: list[str] = []
    for key in sorted(by_stratum.keys()):
        ids = sorted(by_stratum[key])
        rng.shuffle(ids)
        n_holdout = max(1, round(len(ids) * frac))
        holdout.extend(ids[:n_holdout])
        train.extend(ids[n_holdout:])

    return train, holdout
