"""FeedbackScorer: adjusts source_type ranking weights based on install conversion."""
from __future__ import annotations
import statistics
from datetime import datetime
from content_engine.pipeline.states import State
from content_engine.store.config_store import ConfigStore
from content_engine.store.episode_store import EpisodeStore


_DECAY = 0.7
_BOOST = 1.3


class FeedbackScorer:
    def __init__(self, episode_store: EpisodeStore, config_store: ConfigStore):
        self._es = episode_store
        self._cs = config_store

    def update_weights(self, since: datetime) -> int:
        measured = self._es.list_by_state(State.MEASURED)
        in_window = [e for e in measured if e.created_at >= since and e.analytics is not None]
        if not in_window:
            current = self._cs.get("ranking_weights")
            return current.version if current else 0

        per_type: dict[str, list[float]] = {}
        for ep in in_window:
            views = (ep.analytics or {}).get("views") or 0
            installs = (ep.analytics or {}).get("installs") or 0
            if views == 0:
                continue
            per_type.setdefault(ep.source_type, []).append(installs / views)

        if not per_type:
            current = self._cs.get("ranking_weights")
            return current.version if current else 0

        avg_per_type = {st: statistics.mean(rs) for st, rs in per_type.items()}
        median = statistics.median(avg_per_type.values())

        current_row = self._cs.get("ranking_weights")
        current_weights = dict(current_row.value) if current_row else {}

        for st, conv in avg_per_type.items():
            base = current_weights.get(st, 1.0)
            current_weights[st] = base * (_BOOST if conv >= median else _DECAY)

        return self._cs.create_version("ranking_weights", current_weights)
