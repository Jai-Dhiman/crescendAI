# apps/evals/teaching_knowledge/ablation/run_ablation.py
"""4-condition signal ablation orchestrator."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Protocol

from teaching_knowledge.ablation.corrupt_signals import corrupt

CONDITIONS = ("real", "shuffle", "marginal", "flip")
SCALER_MEAN = {
    "dynamics": 0.545, "timing": 0.4848, "pedaling": 0.4594,
    "articulation": 0.5369, "phrasing": 0.5188, "interpretation": 0.5064,
}

JUDGE_SYSTEM = "You are a careful evaluator. Score the teacher synthesis on 7 pedagogical dimensions. Output a JSON array."


class SynthesisClient(Protocol):
    def complete(self, *, user: str, system: str, max_tokens: int) -> str: ...


class JudgeClient(Protocol):
    def complete(self, *, user: str, system: str, max_tokens: int) -> str: ...


SYNTHESIS_SYSTEM = (
    "You are a warm, perceptive piano teacher reviewing a practice session. "
    "Respond in 3-6 sentences."
)


def _muq_to_top_moments(muq_means: dict[str, float]) -> list[dict]:
    moments = []
    for dim, score in muq_means.items():
        dev = score - SCALER_MEAN.get(dim, 0.5)
        if abs(dev) >= 0.05:
            moments.append({
                "dimension": dim, "score": score,
                "deviation_from_mean": round(dev, 3),
                "direction": "above_average" if dev > 0 else "below_average",
            })
    moments.sort(key=lambda m: abs(m["deviation_from_mean"]), reverse=True)
    return moments[:4]


def _build_user_msg(top_moments, duration_seconds, meta) -> str:
    payload = {
        "duration_minutes": round(duration_seconds / 60, 1),
        "practice_pattern": "continuous_play",
        "top_moments": top_moments,
        "drilling_records": [],
        "piece": {"title": meta["title"], "composer": meta["composer"], "skill_level": meta["skill_bucket"]},
    }
    return f"<session_data>\n{json.dumps(payload, indent=2)}\n</session_data>\n<task>Write 3-6 sentences.</task>"


def _call_judge(judge_client: JudgeClient, synthesis_text: str, meta: dict) -> list[dict]:
    """Call the judge and return parsed dimension list. Raises on parse failure."""
    user = (
        f"Piece: {meta['title']} by {meta['composer']}\n"
        f"Student skill level: {meta['skill_bucket']}\n\n"
        f"## AI Teacher Output to Evaluate\n{synthesis_text}"
    )
    raw = judge_client.complete(user=user, system=JUDGE_SYSTEM, max_tokens=4000)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge returned non-JSON: {exc}; raw={raw[:200]!r}") from exc
    if not isinstance(data, list):
        raise ValueError(f"judge response is not a list: raw={raw[:200]!r}")
    return data


def _load_completed(out_path: Path) -> set[tuple[str, str]]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        done.add((row["recording_id"], row["condition"]))
    return done


def run_ablation(
    *,
    sessions: list[dict],
    out_path: Path,
    synthesis_client: SynthesisClient,
    judge_client: JudgeClient | None = None,
    seed: int = 42,
    skip_judge: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(out_path)
    all_real_top_moments = [_muq_to_top_moments(s["muq_means"]) for s in sessions]

    with out_path.open("a") as fout:
        for idx, session in enumerate(sessions):
            real_tm = all_real_top_moments[idx]
            for condition in CONDITIONS:
                key = (session["recording_id"], condition)
                if key in completed:
                    continue
                if condition == "real":
                    used_tm = real_tm
                else:
                    corpus = all_real_top_moments
                    if len(corpus) < 2:
                        corpus = corpus + [list(corpus[0])]
                    used_tm = corrupt(real_tm, mode=condition, seed=seed + idx,
                                      all_top_moments=corpus)
                user_msg = _build_user_msg(used_tm, session["duration_seconds"], session["meta"])
                t0 = time.monotonic()
                synth = synthesis_client.complete(user=user_msg, system=SYNTHESIS_SYSTEM, max_tokens=1024)
                lat = round((time.monotonic() - t0) * 1000)

                judge_dimensions: list[dict] = []
                if not skip_judge and judge_client is not None:
                    judge_dimensions = _call_judge(judge_client, synth, session["meta"])

                row = {
                    "recording_id": session["recording_id"],
                    "condition": condition,
                    "top_moments_used": used_tm,
                    "synthesis_text": synth,
                    "synthesis_latency_ms": lat,
                    "judge_dimensions": judge_dimensions,
                    "judge_skipped": skip_judge or judge_client is None,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
