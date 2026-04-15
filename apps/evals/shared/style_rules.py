from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_STYLE_RULES_PATH = Path(__file__).parent / "data" / "style_rules.json"


@lru_cache(maxsize=1)
def _load_style_rules() -> dict:
    return json.loads(_STYLE_RULES_PATH.read_text())


def composer_to_era(composer: str) -> str:
    """Map a composer name to its stylistic era.

    Uses substring matching against composer_patterns in style_rules.json.
    Returns "Unknown" if no pattern matches.
    """
    if not composer:
        return "Unknown"
    rules = _load_style_rules()
    for era_name, era_data in rules["eras"].items():
        for pattern in era_data["composer_patterns"]:
            if pattern.lower() in composer.lower():
                return era_name
    return "Unknown"


def get_style_guidance(composer: str) -> str:
    """Return an XML-wrapped prose block with per-dimension style guidance.

    Returns an empty string for unknown composers so the caller can omit
    the section entirely from the prompt.
    """
    era = composer_to_era(composer)
    if era == "Unknown":
        return ""
    rules = _load_style_rules()
    dimensions = rules["eras"][era]["dimensions"]
    lines = [
        f'<style_guidance era="{era}">',
        f"For {era}-era repertoire, weight dimensions as follows when giving feedback:",
    ]
    for dim, rule in dimensions.items():
        lines.append(f"- {dim}: {rule}")
    lines.append("Advice that contradicts these rules should not be given.")
    lines.append("</style_guidance>")
    return "\n".join(lines)
