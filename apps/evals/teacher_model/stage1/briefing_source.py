import json
import random
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


def iter_chat_scenarios(template: dict, n: int, seed: int) -> Iterator[Briefing]:
    rng = random.Random(seed)
    intents = template["intents"]
    fillers = template["fillers"]
    for i in range(n):
        intent = rng.choice(intents)
        chosen = {k: rng.choice(v) for k, v in fillers.items()}
        user_text = intent["user"].format(**chosen)
        briefing_id = f"chat_{seed}_{i:04d}_{intent['id']}"
        yield Briefing(
            briefing_id=briefing_id,
            framing_text=user_text,
            composer=str(chosen.get("composer", "")),
            skill_bucket="intermediate",
            shape="chat",
        )


def iter_synthesis_briefings(cache_dir: Path) -> Iterator[Briefing]:
    for path in sorted(cache_dir.glob("*.json")):
        data = json.loads(path.read_text())
        yield Briefing(
            briefing_id=data["briefing_id"],
            framing_text=data["framing_text"],
            composer=data["composer"],
            skill_bucket=data["skill_bucket"],
        )
