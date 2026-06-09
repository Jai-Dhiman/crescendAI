"""
Orchestrator: inference cache -> Sonnet synthesis -> Gemma judge -> JSONL results.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.run_eval
    uv run python -m teaching_knowledge.run_eval --limit 50
    uv run python -m teaching_knowledge.run_eval --out results/my_run.jsonl
    uv run python -m teaching_knowledge.run_eval --dry-run   # synthesis only, skip judge

Models:
    Synthesis: claude-sonnet-4-6 (Anthropic, prod model)
    Judge:     @cf/google/gemma-4-26b-a4b-it (Workers AI)

Resume-safe: any recording_id already written to the output JSONL is skipped.
Errors per recording are written to the JSONL and the run continues.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from shared.provenance import RunProvenance, make_run_provenance

# Root paths
EVALS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALS_ROOT.parents[1]
CACHE_DIR = REPO_ROOT / "model" / "data" / "eval" / "inference_cache" / "auto-t5_http"
SKILL_EVAL_DIR = REPO_ROOT / "model" / "data" / "evals" / "skill_eval"
RESULTS_DIR = EVALS_ROOT / "results"


def _judge_provider_for(judge_model: str) -> str:
    """Autodetect the judge provider from the model name.

    @cf/* => workers-ai; vendor/model => openrouter (bare names unreachable
    here -- the family guard blocks them upstream).
    """
    return "openrouter" if "/" in judge_model and not judge_model.startswith("@cf/") else "workers-ai"


def load_manifests() -> dict[str, dict[str, Any]]:
    """Build a video_id -> metadata lookup from all skill_eval manifests."""
    lookup: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(SKILL_EVAL_DIR.glob("*/manifest.yaml")):
        data = yaml.safe_load(manifest_path.read_text())
        piece_slug = data.get("piece", manifest_path.parent.name)
        title = data.get("title", piece_slug)
        composer = data.get("composer", "Unknown")
        for rec in data.get("recordings", []):
            video_id = rec.get("video_id")
            if video_id:
                lookup[video_id] = {
                    "piece_slug": piece_slug,
                    "title": title,
                    "composer": composer,
                    "skill_bucket": rec.get("skill_bucket", 3),
                }
    return lookup


def load_completed_ids(out_path: Path) -> set[str]:
    """Return recording_ids already written to the output JSONL."""
    if not out_path.exists():
        return set()
    completed: set[str] = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            rid = row.get("recording_id")
            if rid and not row.get("error") and row.get("synthesis_text"):
                completed.add(rid)
        except json.JSONDecodeError:
            pass
    return completed


def render_artifact_text(artifact: dict[str, Any] | None) -> str | None:
    """Render the full V6 SynthesisArtifact into the prose the judge scores.

    The live WS `text` carries only `headline` (the student's lead-in). The eval must
    grade the COMPLETE V6 teaching output, so we render the headline plus the focus
    areas, strengths, and proposed exercises that the headline omits (#28). Returns
    None when no structured artifact is present (caller falls back to the headline).

    Mirrors the TS shape attached in session-brain.ts buildEvalContext: keys
    `headline`, `focus_areas` [{dimension, one_liner, severity}], `strengths`
    [{dimension, one_liner}], `proposed_exercises` [str].
    """
    if not artifact:
        return None

    sections: list[str] = []

    headline = (artifact.get("headline") or "").strip()
    if headline:
        sections.append(headline)

    focus_areas = artifact.get("focus_areas") or []
    if focus_areas:
        lines = ["Focus areas:"]
        for fa in focus_areas:
            severity = fa.get("severity", "")
            dimension = fa.get("dimension", "")
            one_liner = (fa.get("one_liner") or "").strip()
            prefix = f"[{severity}] " if severity else ""
            lines.append(f"- {prefix}{dimension}: {one_liner}")
        sections.append("\n".join(lines))

    strengths = artifact.get("strengths") or []
    if strengths:
        lines = ["Strengths:"]
        for st in strengths:
            dimension = st.get("dimension", "")
            one_liner = (st.get("one_liner") or "").strip()
            lines.append(f"- {dimension}: {one_liner}")
        sections.append("\n".join(lines))

    exercises = artifact.get("proposed_exercises") or []
    if exercises:
        lines = ["Suggested exercises:"]
        for ex in exercises:
            ex_text = (ex or "").strip()
            if ex_text:
                lines.append(f"- {ex_text}")
        if len(lines) > 1:
            sections.append("\n".join(lines))

    rendered = "\n\n".join(sections).strip()
    return rendered or None


def build_do_row(
    session_result,
    meta: dict[str, Any],
    judge_fn,
    provenance: RunProvenance,
    *,
    dry_run: bool = False,
    judge_provider: str = "workers-ai",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
) -> dict:
    """Build one aggregator-schema JSONL row from a DO SessionResult.

    No thin-framing fallback: a DO/WS failure (or missing synthesis) records the
    error verbatim and skips the judge. Piece resolution is reported via
    `piece_resolved` (False -> the DO ran Tier 2/3 with bar_range null).
    """
    recording_id = getattr(session_result, "recording_id", meta.get("recording_id", ""))
    piece_resolved = getattr(session_result, "piece_identification", None) is not None

    base = {
        "recording_id": recording_id,
        "run_id": provenance.run_id,
        "git_sha": provenance.git_sha,
        "git_dirty": provenance.git_dirty,
        "piece_slug": meta.get("piece_slug", ""),
        "title": meta.get("title", ""),
        "composer": meta.get("composer", ""),
        "skill_bucket": meta.get("skill_bucket", 3),
        "piece_resolved": piece_resolved,
        "synthesis_text": "",
        "synthesis_latency_ms": int(getattr(session_result, "synthesis_latency_ms", 0) or 0),
        "judge_dimensions": [],
        "judge_model": "",
        "judge_latency_ms": 0,
        "error": "",
    }

    synthesis = getattr(session_result, "synthesis", None)
    errors = getattr(session_result, "errors", []) or []
    if synthesis is None or not getattr(synthesis, "text", ""):
        base["error"] = "; ".join(errors) if errors else "DO returned no synthesis text"
        return base

    # Judge the COMPLETE V6 teaching output, not the headline alone. The DO attaches
    # the full structured artifact under eval_context.artifact (#28); render it to prose.
    # Fall back to the headline only when no artifact is present (e.g. an older payload).
    eval_context = getattr(synthesis, "eval_context", {}) or {}
    artifact = eval_context.get("artifact")
    judged_text = render_artifact_text(artifact) or synthesis.text
    base["synthesis_text"] = judged_text

    if dry_run:
        base["judge_model"] = "dry_run"
        return base

    judge_ctx = {
        "piece_name": meta.get("title", ""),
        "composer": meta.get("composer", ""),
        "skill_level": meta.get("skill_bucket", 3),
    }
    jr = judge_fn(
        judged_text,
        judge_ctx,
        provider=judge_provider,
        model=judge_model,
    )
    base["judge_dimensions"] = [
        {
            "criterion": d.criterion,
            "process": getattr(d, "process", None),
            "outcome": getattr(d, "outcome", None),
            "score": getattr(d, "score", None),
            "evidence": getattr(d, "evidence", ""),
            "reason": getattr(d, "reason", ""),
        }
        for d in jr.dimensions
    ]
    base["judge_model"] = getattr(jr, "model", "")
    base["judge_latency_ms"] = round(getattr(jr, "latency_ms", 0.0))
    return base


def _default_driver(wrangler_url, recording_cache, student_id, piece_query):
    """Default DO-replay driver: real run_recording over wrangler dev."""
    from shared.pipeline_client import run_recording

    return asyncio.run(
        run_recording(
            wrangler_url,
            recording_cache,
            student_id=student_id,
            piece_query=piece_query,
        )
    )


def run_do_baseline(
    holdout_path: Path,
    out_path: Path,
    wrangler_url: str,
    judge_fn,
    *,
    driver=_default_driver,
    limit: int | None = None,
    dry_run: bool = False,
    judge_provider: str = "workers-ai",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
) -> None:
    """Drive holdout recordings through the real SessionBrain DO and write JSONL.

    `driver(wrangler_url, recording_cache, student_id, piece_query) -> SessionResult`
    is injected so the orchestration is unit-testable without a live DO.
    The default driver calls shared.pipeline_client.run_recording.
    """
    holdout = [
        json.loads(line)
        for line in holdout_path.read_text().splitlines()
        if line.strip()
    ]
    if limit is not None:
        holdout = holdout[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(out_path)
    provenance = make_run_provenance()
    print(f"run_id: {provenance.run_id} | DO path | wrangler={wrangler_url}")

    processed = 0
    errors = 0

    with out_path.open("a") as fout:
        for entry in holdout:
            recording_id = entry["recording_id"]
            if recording_id in completed:
                continue

            meta = {
                "recording_id": recording_id,
                "piece_slug": entry.get("piece_slug", ""),
                "title": entry.get("title", ""),
                "composer": entry.get("composer", ""),
                "skill_bucket": int(entry.get("skill_bucket", 3)),
            }

            briefing = json.loads(Path(entry["briefing_path"]).read_text())
            recording_cache = {
                "recording_id": briefing.get("recording_id", recording_id),
                "chunks": briefing.get("chunks", []),
            }

            # Each recording gets a fresh per-recording eval identity so
            # sessionHistory / pastDiagnoses queries in the DO see an empty
            # longitudinal record (cold-start fidelity).
            per_recording_student_id = f"eval-{recording_id}"

            try:
                session_result = driver(
                    wrangler_url,
                    recording_cache,
                    per_recording_student_id,
                    meta["piece_slug"] or None,
                )
                row = build_do_row(
                    session_result,
                    meta,
                    judge_fn,
                    provenance,
                    dry_run=dry_run,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                )
            except Exception as exc:  # driver/judge hard failure -> error row, never thin fallback
                row = {
                    "recording_id": recording_id,
                    "run_id": provenance.run_id,
                    "git_sha": provenance.git_sha,
                    "git_dirty": provenance.git_dirty,
                    "piece_slug": meta["piece_slug"],
                    "title": meta["title"],
                    "composer": meta["composer"],
                    "skill_bucket": meta["skill_bucket"],
                    "piece_resolved": False,
                    "synthesis_text": "",
                    "synthesis_latency_ms": 0,
                    "judge_dimensions": [],
                    "judge_model": "",
                    "judge_latency_ms": 0,
                    "error": str(exc)[:500],
                }

            if row.get("error"):
                errors += 1
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            processed += 1
            print(f"[{processed}] {recording_id} | {meta['piece_slug']}"
                  + (f" | ERROR: {row['error'][:80]}" if row.get("error") else ""))

    print(f"Done. processed={processed} errors={errors} -> {out_path}")


def main() -> None:
    """Run the teaching-quality eval through the real SessionBrain V6 DO.

    The DO replay is the ONLY path: it drives each holdout recording's cached
    inference through wrangler dev and judges the full V6 SynthesisArtifact
    (#28). The legacy Python-framed `run()` synthesis path is gone.
    """
    parser = argparse.ArgumentParser(
        description="Teaching quality eval runner (V6 DO path)."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max recordings to process (default: all)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSONL path (default: results/baseline_v2_do.jsonl)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run synthesis through the DO but skip the judge (smoke check).",
    )
    parser.add_argument(
        "--judge-model", default="@cf/google/gemma-4-26b-a4b-it",
        help="Judge model name (default: @cf/google/gemma-4-26b-a4b-it)",
    )
    parser.add_argument(
        "--wrangler-url", default="http://localhost:8787",
        help="wrangler dev base URL (default: %(default)s).",
    )
    args = parser.parse_args()

    from shared.judge import judge_synthesis_v2

    judge_provider = _judge_provider_for(args.judge_model)
    holdout = EVALS_ROOT / "teacher_model" / "stage0" / "data" / "stage0_holdout.jsonl"
    out = args.out if args.out is not None else (RESULTS_DIR / "baseline_v2_do.jsonl")
    run_do_baseline(
        holdout_path=holdout,
        out_path=out,
        wrangler_url=args.wrangler_url,
        judge_fn=judge_synthesis_v2,
        limit=args.limit,
        dry_run=args.dry_run,
        judge_provider=judge_provider,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    main()
