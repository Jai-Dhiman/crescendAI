"""Orchestrator: dumb state-machine dispatcher.

Exception policy: each handler narrows its `except` to the specific failure types
its dependency can raise. Unknown exceptions propagate so Sentry sees them and
so the engine never silently masks a bug behind a FAILED_* marker. Per spec
'explicit exception handling, no silent fallbacks'.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore
from content_engine.store.config_store import ConfigStore
from content_engine.adapters.model_runner import ModelOutput, InferenceError
from content_engine.adapters.llm_gateway import LlmGatewayError
from content_engine.adapters.scheduler import SchedulerError
from content_engine.agents.observation_selector import (
    ClipMetadata,
    Observation,
    ObservationSelectorError,
)
from content_engine.agents.critic_truthfulness import CriticTruthfulnessError
from content_engine.render.renderer import RendererError
from content_engine.render.templates import CtaTemplate


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
        config_store: ConfigStore | None = None,
        cross_post_platforms: list[str] | None = None,
    ):
        self._es = episode_store
        self._model = model_runner
        self._clips = clip_paths
        self._obs = observation_selector
        self._nar = narrator
        self._crit = critic
        self._ren = renderer
        self._sched = scheduler
        self._cs = config_store
        self._platforms = cross_post_platforms or ["youtube", "tiktok", "instagram"]

    def tick(self) -> None:
        # Snapshot all work queues before any handler runs so a transition in
        # one state cannot cause the same episode to be processed twice per tick.
        # FAILED_* states are intentional dead-ends for MVP: a failed episode
        # requires manual DB intervention or a future /episode/<id>/retry endpoint.
        # Transient failures (e.g. InferenceError) will surface via Sentry.
        work = [
            (self._es.list_by_state(State.CURATED), self._handle_curated),
            (self._es.list_by_state(State.ANALYZED), self._handle_analyzed),
            (self._es.list_by_state(State.OBSERVATION_SELECTED), self._handle_observation_selected),
            (self._es.list_by_state(State.SCRIPT_DRAFTED), self._handle_script_drafted),
            (self._es.list_by_state(State.RECORDED), self._handle_recorded),
            (self._es.list_by_state(State.RENDERED), self._handle_rendered),
        ]
        for episodes, handler in work:
            for ep in episodes:
                handler(ep)

    def _cta(self) -> CtaTemplate:
        if self._cs is None:
            return CtaTemplate.for_phase("A")
        row = self._cs.get("cta")
        phase = row.value.get("phase", "A") if row else "A"
        return CtaTemplate.for_phase(phase)

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
        ep.model_output = {"scores": output.scores, "duration_sec": output.duration_sec}
        self._es.save(ep)
        self._es.transition(ep.id, State.ANALYZED)

    def _handle_analyzed(self, ep: Episode) -> None:
        if self._obs is None or ep.model_output is None:
            self._es.transition(ep.id, State.FAILED_OBSERVATION)
            return
        try:
            mo = ModelOutput(scores=ep.model_output["scores"], duration_sec=ep.model_output["duration_sec"], raw={})
            obs = self._obs.select(mo, ClipMetadata(duration_sec=mo.duration_sec))
        except (ObservationSelectorError, LlmGatewayError):
            self._es.transition(ep.id, State.FAILED_OBSERVATION)
            return
        ep.observation = {"dimension": obs.dimension, "time_range": list(obs.time_range), "plain_english": obs.plain_english}
        self._es.save(ep)
        self._es.transition(ep.id, State.OBSERVATION_SELECTED)

    def _handle_observation_selected(self, ep: Episode) -> None:
        if self._nar is None or ep.observation is None:
            self._es.transition(ep.id, State.FAILED_SCRIPT)
            return
        try:
            obs = Observation(
                dimension=ep.observation["dimension"],
                time_range=tuple(ep.observation["time_range"]),
                plain_english=ep.observation["plain_english"],
            )
            script = self._nar.write_script(obs, self._cta(), style_examples=[])
        except LlmGatewayError:
            self._es.transition(ep.id, State.FAILED_SCRIPT)
            return
        ep.script_text = script.text
        self._es.save(ep)
        self._es.transition(ep.id, State.SCRIPT_DRAFTED)

    def _handle_script_drafted(self, ep: Episode) -> None:
        if self._crit is None or ep.observation is None:
            self._es.transition(ep.id, State.FAILED_CRITIC)
            return
        clip = self._clips.get(ep.id)
        if clip is None:
            self._es.transition(ep.id, State.FAILED_CRITIC)
            return
        try:
            obs = Observation(
                dimension=ep.observation["dimension"],
                time_range=tuple(ep.observation["time_range"]),
                plain_english=ep.observation["plain_english"],
            )
            verdict = self._crit.verify(clip, obs)
        except (CriticTruthfulnessError, LlmGatewayError):
            self._es.transition(ep.id, State.FAILED_CRITIC)
            return
        if verdict.passed:
            self._es.transition(ep.id, State.CRITIC_PASSED)
        else:
            self._es.transition(ep.id, State.KILLED_TRUTHFULNESS)

    def _handle_recorded(self, ep: Episode) -> None:
        if self._ren is None:
            self._es.transition(ep.id, State.FAILED_RENDER)
            return
        try:
            path = self._ren.render(ep, self._cta())
        except RendererError:
            self._es.transition(ep.id, State.FAILED_RENDER)
            return
        ep.render_path = str(path)
        self._es.save(ep)
        self._es.transition(ep.id, State.RENDERED)

    def _handle_rendered(self, ep: Episode) -> None:
        if self._sched is None or ep.render_path is None:
            self._es.transition(ep.id, State.FAILED_SCHEDULE)
            return
        try:
            results = self._sched.schedule(
                asset_path=Path(ep.render_path),
                when=datetime.now(timezone.utc),
                platforms=self._platforms,
                caption=(ep.script_text or "")[:120],
                description_link=self._cta().landing_url + "?utm_source=shorts&utm_medium=organic&utm_campaign=ce",
            )
        except SchedulerError:
            self._es.transition(ep.id, State.FAILED_SCHEDULE)
            return
        ep.posts = {r.platform: r.post_id for r in results if r.post_id is not None}
        self._es.save(ep)
        self._es.transition(ep.id, State.SCHEDULED)
