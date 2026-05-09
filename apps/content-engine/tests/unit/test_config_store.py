"""Verifies versioned ConfigStore: immutability + latest resolution."""
from content_engine.store.config_store import ConfigStore


def test_create_then_get_returns_latest(tmp_path):
    store = ConfigStore(db_path=tmp_path / "c.sqlite")
    v1 = store.create_version("cta", {"phase": "A"})
    v2 = store.create_version("cta", {"phase": "B"})
    assert v2 > v1
    cfg = store.get("cta")
    assert cfg.value == {"phase": "B"}
    assert cfg.version == v2


def test_get_specific_version_returns_historical_value(tmp_path):
    store = ConfigStore(db_path=tmp_path / "c.sqlite")
    v1 = store.create_version("cta", {"phase": "A"})
    store.create_version("cta", {"phase": "B"})
    cfg = store.get("cta", version=v1)
    assert cfg.value == {"phase": "A"}


def test_get_unknown_key_returns_none(tmp_path):
    store = ConfigStore(db_path=tmp_path / "c.sqlite")
    assert store.get("never_set") is None
