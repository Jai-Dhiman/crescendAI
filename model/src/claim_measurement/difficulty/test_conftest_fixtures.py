"""#175: the fixture builder every ensemble test consumes, tested itself."""
import numpy as np

from claim_measurement.difficulty.score_wav import RidgeHead, read_ridge_head


def _head(dim=4, coef=1.0, fallback=5.0):
    return RidgeHead(mean=np.zeros(dim), scale=np.ones(dim),
                     coef=np.full(dim, coef), intercept=0.0,
                     fallback_score=fallback)


def test_the_fixture_writes_a_readable_two_member_directory(make_model_dir):
    model_dir = make_model_dir({"s2026": _head(coef=1.0),
                                "s2027": _head(coef=2.0)})

    for name, expected in (("s2026", 1.0), ("s2027", 2.0)):
        member = model_dir / "members" / name
        assert (member / "adapter" / "adapter_config.json").exists()
        np.testing.assert_array_equal(
            read_ridge_head(member / "ridge_head.npz").coef, np.full(4, expected))


def test_the_fixture_can_also_write_the_legacy_single_adapter_layout(make_model_dir):
    model_dir = make_model_dir({"only": _head()}, layout="legacy")

    assert (model_dir / "adapter" / "adapter_config.json").exists()
    assert (model_dir / "ridge_head.npz").exists()
    assert not (model_dir / "members").exists()
