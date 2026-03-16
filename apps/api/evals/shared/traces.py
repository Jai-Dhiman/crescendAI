from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineTrace:
    observation_id: str
    recording_id: str
    chunk_index: int
    inference: dict[str, Any] = field(default_factory=dict)
    stop_score: float = 0.0
    stop_triggered: bool = False
    teaching_moment: dict[str, Any] = field(default_factory=dict)
    analysis_facts: dict[str, Any] = field(default_factory=dict)
    subagent_output: dict[str, Any] = field(default_factory=dict)
    teacher_observation: str = ""
    judge_scores: list[dict[str, Any]] = field(default_factory=list)

    def save(self, traces_dir: str | Path) -> Path:
        """Write trace as JSON to traces_dir. Returns the written path."""
        traces_dir = Path(traces_dir)
        traces_dir.mkdir(parents=True, exist_ok=True)
        path = traces_dir / f"{self.observation_id}.json"
        data = {
            "observation_id": self.observation_id,
            "recording_id": self.recording_id,
            "chunk_index": self.chunk_index,
            "inference": self.inference,
            "stop_score": self.stop_score,
            "stop_triggered": self.stop_triggered,
            "teaching_moment": self.teaching_moment,
            "analysis_facts": self.analysis_facts,
            "subagent_output": self.subagent_output,
            "teacher_observation": self.teacher_observation,
            "judge_scores": self.judge_scores,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path


def load_trace(traces_dir: str | Path, observation_id: str) -> PipelineTrace:
    """Load a trace by observation_id from the traces directory."""
    traces_dir = Path(traces_dir)
    path = traces_dir / f"{observation_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Trace not found: {path}")

    data = json.loads(path.read_text())
    return PipelineTrace(
        observation_id=data["observation_id"],
        recording_id=data["recording_id"],
        chunk_index=data["chunk_index"],
        inference=data.get("inference", {}),
        stop_score=data.get("stop_score", 0.0),
        stop_triggered=data.get("stop_triggered", False),
        teaching_moment=data.get("teaching_moment", {}),
        analysis_facts=data.get("analysis_facts", {}),
        subagent_output=data.get("subagent_output", {}),
        teacher_observation=data.get("teacher_observation", ""),
        judge_scores=data.get("judge_scores", []),
    )
