"""Verifies project is installable and importable."""

def test_package_imports():
    import content_engine
    assert content_engine.__name__ == "content_engine"
