from __future__ import annotations

from shared.style_rules import composer_to_era, get_style_guidance


def test_bach_maps_to_baroque() -> None:
    assert composer_to_era("Johann Sebastian Bach") == "Baroque"
    assert composer_to_era("Bach") == "Baroque"


def test_chopin_maps_to_romantic() -> None:
    assert composer_to_era("Frederic Chopin") == "Romantic"
    assert composer_to_era("Chopin") == "Romantic"


def test_debussy_maps_to_impressionist() -> None:
    assert composer_to_era("Claude Debussy") == "Impressionist"


def test_mozart_maps_to_classical() -> None:
    assert composer_to_era("Wolfgang Amadeus Mozart") == "Classical"


def test_unknown_composer_returns_unknown_era() -> None:
    assert composer_to_era("Unknown") == "Unknown"
    assert composer_to_era("") == "Unknown"
    assert composer_to_era("Xyz Fake Composer") == "Unknown"


def test_style_guidance_for_bach_mentions_articulation() -> None:
    guidance = get_style_guidance("Bach")
    assert "Baroque" in guidance
    assert "articulation" in guidance.lower()
    assert "pedaling" in guidance.lower()


def test_style_guidance_for_chopin_mentions_dynamics_and_pedaling() -> None:
    guidance = get_style_guidance("Chopin")
    assert "Romantic" in guidance
    assert "dynamics" in guidance.lower()
    assert "pedaling" in guidance.lower()


def test_style_guidance_for_unknown_returns_empty_string() -> None:
    assert get_style_guidance("Unknown") == ""
    assert get_style_guidance("") == ""


def test_style_guidance_is_formatted_as_xml_block() -> None:
    guidance = get_style_guidance("Bach")
    assert guidance.startswith("<style_guidance")
    assert guidance.endswith("</style_guidance>")
