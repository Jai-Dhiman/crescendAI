"""Orchestrator: dumb state-machine dispatcher.

Reads pending episodes per state, calls the right component, persists transitions.
No business logic lives here.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore
from content_engine.adapters.model_runner import InferenceError


class Orchestrator:
    def __init__(
        self,
        episode_store: EpisodeStore,
        model_runner: Any,
        clip_paths: dict[str, Path],
        observation_selector: Any,
        narrator: Any,
        critic: Any,
        renderer: Any,
        scheduler: Any,
    ):
        self._es = episode_store
        self._model = model_runner
        self._clips = clip_paths
        self._obs = observation_selector
        self._nar = narrator
        self._crit = critic
        self._ren = renderer
        self._sched = scheduler

    def tick(self) -> None:
        for ep in self._es.list_by_state(State.CURATED):
            self._handle_curated(ep)

    def _handle_curated(self, ep: Episode) -> None:
        clip = self._clips.get(ep.id)
        if clip is None:
            self._es.transition(ep.id, State.FAILED_ANALYSIS)
            return
        try:
            output = self._model.run(clip)
        except InferenceError:
            self._es.transition(ep.id, State.FAILED_ANALYSIS)
            return
        ep.model_output = {
            "scores": output.scores,
            "duration_sec": output.duration_sec,
        }
        self._es.save(ep)
        self._es.transition(ep.id, State.ANALYZED)
