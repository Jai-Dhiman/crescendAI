# model/src/piece_id_eval/test_parity_fixtures.py
import json
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parents[2] / "data/evals/piece_id/parity_fixtures.json"


def test_parity_fixtures_reproduce_certified_operating_point():
    data = json.loads(_FIXTURES.read_text())
    assert data["margin_threshold"] == 0.0935
    queries = data["queries"]

    in_cat = [q for q in queries if q["in_catalog"]]
    ood = [q for q in queries if not q["in_catalog"]]
    assert len(in_cat) == 16, f"expected 16 in-catalog queries, got {len(in_cat)}"
    assert len(ood) >= 10, f"expected >=10 OOD queries, got {len(ood)}"

    # Certified TA point estimate is 0.875 -> >=14/16 in-catalog queries lock.
    assert sum(q["expected_locked"] for q in in_cat) >= 14
    # Certified false-accept axis: OOD queries must not lock.
    assert sum(q["expected_locked"] for q in ood) == 0

    for q in queries:
        assert len(q["query_events"]) >= 2
        assert 2 <= len(q["candidates"]) <= 5
        for c in q["candidates"]:
            assert all(0 <= m <= 0x0FFF for m in c["events"])  # 12-bit pc masks
        # margin is (2nd-best - best) over candidate costs
        costs = sorted(c["expected_cost"] for c in q["candidates"])
        assert abs(q["expected_margin"] - (costs[1] - costs[0])) < 1e-9
        assert q["expected_locked"] == (q["expected_margin"] >= 0.0935)
