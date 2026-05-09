from teacher_model.stage1.briefing_source import Briefing
from teacher_model.stage1.holdout import split_holdout


def _make_pool() -> list[Briefing]:
    out = []
    for composer in ("Chopin", "Bach", "Mozart", "Debussy"):
        for skill in ("beginner", "intermediate", "advanced"):
            for i in range(20):  # 20 per stratum, 240 total
                out.append(
                    Briefing(
                        briefing_id=f"{composer}_{skill}_{i:03d}",
                        framing_text="stub",
                        composer=composer,
                        skill_bucket=skill,
                    )
                )
    return out


def test_split_holdout_proportional_per_stratum_and_disjoint():
    pool = _make_pool()
    train, holdout = split_holdout(
        briefings=pool, frac=0.10, strata=["composer", "skill_bucket"], seed=42
    )

    # disjoint
    assert set(train).isdisjoint(set(holdout))
    # union covers pool
    assert set(train) | set(holdout) == {b.briefing_id for b in pool}
    # within +-1 of 10% per stratum (20 per stratum -> ~2 in holdout)
    for composer in ("Chopin", "Bach", "Mozart", "Debussy"):
        for skill in ("beginner", "intermediate", "advanced"):
            stratum_in_holdout = [
                bid for bid in holdout if bid.startswith(f"{composer}_{skill}_")
            ]
            assert 1 <= len(stratum_in_holdout) <= 3, (
                composer,
                skill,
                len(stratum_in_holdout),
            )


def test_split_holdout_deterministic_under_same_seed():
    pool = _make_pool()
    a = split_holdout(pool, 0.10, ["composer", "skill_bucket"], seed=42)
    b = split_holdout(pool, 0.10, ["composer", "skill_bucket"], seed=42)
    c = split_holdout(pool, 0.10, ["composer", "skill_bucket"], seed=99)
    assert a == b
    assert a != c
