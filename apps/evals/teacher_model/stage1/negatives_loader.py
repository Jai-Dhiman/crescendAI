from pathlib import Path

from pydantic import ValidationError

from teacher_model.stage1.schema import MatchedContrastPair, Stage1Negative


class NegativeLoadError(Exception):
    pass


def load_negatives(dir: Path) -> list[Stage1Negative]:
    loaded: list[Stage1Negative] = []
    for path in sorted(dir.glob("*.json")):
        raw = path.read_text()
        try:
            loaded.append(Stage1Negative.model_validate_json(raw))
        except ValidationError as exc:
            raise NegativeLoadError(
                f"{path.name}: validation failed -- {exc}"
            ) from exc
    return loaded


def load_pairs(dir: Path) -> list[MatchedContrastPair]:
    loaded: list[MatchedContrastPair] = []
    for path in sorted(dir.glob("*.json")):
        raw = path.read_text()
        try:
            loaded.append(MatchedContrastPair.model_validate_json(raw))
        except ValidationError as exc:
            raise NegativeLoadError(
                f"{path.name}: validation failed -- {exc}"
            ) from exc
    return loaded
