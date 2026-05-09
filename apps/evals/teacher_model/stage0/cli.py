"""Stage 0 CLI: pin-tokenizer / sample / synthesis / tool / continuation / mcq / aggregate."""
from __future__ import annotations

import argparse
from pathlib import Path

from teacher_model.stage0.aggregator import build_dossier
from teacher_model.stage0.judge_extended import judge_extended
from teacher_model.stage0.pin_tokenizer import pin_tokenizer
from teacher_model.stage0.run_continuation import run as run_continuation
from teacher_model.stage0.run_synthesis import run as run_synthesis
from teacher_model.stage0.run_tool_probe import run as run_tool_probe
from teacher_model.stage0.sampler import sample_holdout, write_holdout

_STAGE0_ROOT = Path(__file__).parent
_DATA_DIR = _STAGE0_ROOT / "data"
_RESULTS_DIR = _STAGE0_ROOT / "results"
_PROMPTS_DIR = _STAGE0_ROOT / "prompts"
_REPO_ROOT = _STAGE0_ROOT.resolve().parents[4]
_SYNTH_SYSTEM = _REPO_ROOT / "apps" / "shared" / "teacher-style" / "synthesis_system.txt"
_BASELINE_AGGREGATE = _REPO_ROOT / "apps" / "evals" / "results" / "baseline_v1_aggregate.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="teacher_model.stage0")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pin-tokenizer", help="Download + hash the Qwen tokenizer")
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "tokenizer_pin.json")

    sp = sub.add_parser("sample", help="Stratified holdout sampling")
    sp.add_argument("--n", type=int, default=100)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out", type=Path, default=_DATA_DIR / "stage0_holdout.jsonl")

    sp = sub.add_parser("synthesis", help="Pipeline A: synthesis eval")
    sp.add_argument("--provider", default="openrouter")
    sp.add_argument("--model", required=True)
    sp.add_argument("--judge-provider", default="workers-ai")
    sp.add_argument("--judge-model", default="@cf/google/gemma-4-26b-a4b-it")
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "synthesis_runs.jsonl")
    sp.add_argument("--limit", type=int, default=None)

    sp = sub.add_parser("tool", help="Pipeline B: tool-call probe")
    sp.add_argument("--provider", default="openrouter")
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "tool_runs.jsonl")

    sp = sub.add_parser("continuation", help="Pipeline B+: post-tool-result continuation probe")
    sp.add_argument("--provider", default="openrouter")
    sp.add_argument("--model", required=True)
    sp.add_argument("--tool-runs", type=Path, default=_RESULTS_DIR / "tool_runs.jsonl")
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "continuation_runs.jsonl")

    sp = sub.add_parser("mcq", help="Pipeline C: existing 50-Q MCQ")
    sp.add_argument("--provider", default="openrouter")
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", type=Path, default=_RESULTS_DIR / "mcq_summary.json")

    sp = sub.add_parser("aggregate", help="Build the capability dossier")
    sp.add_argument("--out-dir", type=Path, default=_RESULTS_DIR)

    return p


def sample_main(*, briefings_dir: Path, manifests: dict, n: int, seed: int, out_path: Path) -> None:
    rows = sample_holdout(briefings_dir, manifests, n=n, seed=seed)
    write_holdout(rows, out_path)


def _load_tool_schemas() -> dict:
    """Mirror of the 6-tool palette from apps/api/src/services/tool-processor.ts."""
    return {
        "create_exercise": {
            "type": "object",
            "properties": {
                "skill": {"type": "string"},
                "exercises": {"type": "array"},
            },
            "required": ["skill", "exercises"],
        },
        "score_highlight": {
            "type": "object",
            "properties": {
                "highlight_id": {"type": "string"},
                "bars": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["bars"],
        },
        "keyboard_guide": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
        "show_session_data": {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
        "reference_browser": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "search_catalog": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }


def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "pin-tokenizer":
        pin_tokenizer(model_id=args.model, out_path=args.out)
        return

    if args.cmd == "sample":
        raise NotImplementedError(
            "sample subcommand requires load_manifests from teaching_knowledge.run_eval; "
            "run sample_main() directly with a pre-loaded manifests dict."
        )

    if args.cmd == "synthesis":
        from teaching_knowledge.llm_client import LLMClient
        client = LLMClient(provider=args.provider, model=args.model)
        run_synthesis(
            holdout_path=_DATA_DIR / "stage0_holdout.jsonl",
            out_path=args.out,
            teacher_client=client,
            judge_fn=judge_extended,
            system_prompt=_SYNTH_SYSTEM.read_text(),
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
        )
        return

    if args.cmd == "tool":
        from teaching_knowledge.llm_client import LLMClient
        client = LLMClient(provider=args.provider, model=args.model)
        run_tool_probe(
            cases_path=_DATA_DIR / "tool_probe_cases.jsonl",
            system_prompt_path=_PROMPTS_DIR / "tool_probe_system.txt",
            schemas=_load_tool_schemas(),
            teacher_client=client,
            out_path=args.out,
        )
        return

    if args.cmd == "continuation":
        from teaching_knowledge.llm_client import LLMClient
        client = LLMClient(provider=args.provider, model=args.model)
        run_continuation(
            tool_runs_path=args.tool_runs,
            cases_path=_DATA_DIR / "tool_probe_cases.jsonl",
            teacher_client=client,
            out_path=args.out,
        )
        return

    if args.cmd == "mcq":
        import subprocess
        import sys
        cmd = [
            sys.executable, "-m", "teacher_model.domain_knowledge_probe",
            "--provider", args.provider, "--model", args.model,
            "--output", str(args.out),
        ]
        subprocess.run(cmd, check=True)
        return

    if args.cmd == "aggregate":
        cont_path = _RESULTS_DIR / "continuation_runs.jsonl"
        build_dossier(
            synthesis_jsonl=_RESULTS_DIR / "synthesis_runs.jsonl",
            tool_jsonl=_RESULTS_DIR / "tool_runs.jsonl",
            mcq_json=_RESULTS_DIR / "mcq_summary.json",
            baseline_aggregate_json=_BASELINE_AGGREGATE,
            out_dir=args.out_dir,
            continuation_jsonl=cont_path if cont_path.exists() else None,
        )
        return
