import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Briefing:
    briefing_id: str
    framing_text: str
    composer: str
    skill_bucket: str
    shape: str = "synthesis"


def iter_synthesis_briefings(cache_dir: Path) -> Iterator[Briefing]:
    for path in sorted(cache_dir.glob("*.json")):
        data = json.loads(path.read_text())
        yield Briefing(
            briefing_id=data["briefing_id"],
            framing_text=data["framing_text"],
            composer=data.get("composer", ""),
            skill_bucket=data.get("skill_bucket", ""),
        )
