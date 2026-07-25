"""Gate 0 probe CLI: contrast-pair manifest -> per-pair Inkling judgments
-> deterministic report.json + run_meta.json.

Offline mode (--recorded) replays canned responses and never touches the
network; live mode requires the Tinker SDK plus explicit USD token rates
(there is no silent default rate). Responses append to
<run-dir>/responses.jsonl after every call, so a crashed or
budget-stopped run resumes by skipping answered pairs. Tinker API errors
and BudgetExceededError propagate as-is -- saved responses ARE the run
state. The report is only written once every manifest pair has a
response. Exit code: 0 on PASS, 1 on FAIL.

Usage:
    uv run python -m audio_teacher.probe --manifest data/manifests/gate0.yaml \
        --recorded runs/recorded.jsonl                      # offline re-score
    uv run python -m audio_teacher.probe --manifest data/manifests/gate0.yaml \
        --max-spend 50.0 --usd-per-1m-input-tokens X \
        --usd-per-1m-output-tokens Y                        # live (user-run)
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from audio_teacher.budget import BudgetGuard
from audio_teacher.client import ProbeClient, RecordedResponseClient
from audio_teacher.manifest import load_manifest
from audio_teacher.scorer import render_report, score_responses

MODEL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = MODEL_ROOT / "data" / "results" / "audio_teacher"


def _read_responses(path: Path) -> tuple[dict[str, str], float]:
    """Load prior responses and sum their cost_usd for budget carry-forward.

    The writer (main(), below) always emits cost_usd on every record, so a
    record missing it on resume means responses.jsonl is corrupt -- fail
    loudly naming the offending pair rather than silently treating it as
    free.
    """
    responses: dict[str, str] = {}
    spent_usd = 0.0
    if path.exists():
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pair_id = rec["pair_id"]
                responses[pair_id] = rec["text"]
                if "cost_usd" not in rec:
                    raise ValueError(
                        f"responses.jsonl record for pair {pair_id!r} in {path} is "
                        "missing cost_usd -- the writer always emits it, so this "
                        "record is corrupt"
                    )
                spent_usd += rec["cost_usd"]
    return responses, spent_usd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="root that manifest clip paths are relative to (default: model/)",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="default: model/data/results/audio_teacher/<manifest stem>",
    )
    parser.add_argument(
        "--recorded", type=Path, default=None,
        help="offline mode: replay responses from this JSONL instead of Tinker",
    )
    parser.add_argument("--max-spend", type=float, default=50.0)
    parser.add_argument("--usd-per-1m-input-tokens", type=float, default=None)
    parser.add_argument("--usd-per-1m-output-tokens", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    load_kwargs = {}
    if args.repo_root is not None:
        load_kwargs["repo_root"] = args.repo_root
    manifest = load_manifest(args.manifest, **load_kwargs)

    if args.recorded is not None:
        client: ProbeClient = RecordedResponseClient(args.recorded)
    else:
        if args.usd_per_1m_input_tokens is None or args.usd_per_1m_output_tokens is None:
            parser.error(
                "live mode requires --usd-per-1m-input-tokens and "
                "--usd-per-1m-output-tokens (no silent default rate exists; "
                "current rates: see the Gate 0 issue)"
            )
        from audio_teacher.tinker_client import TinkerProbeClient

        client = TinkerProbeClient(
            sample_rate=manifest.sample_rate,
            usd_per_1m_input_tokens=args.usd_per_1m_input_tokens,
            usd_per_1m_output_tokens=args.usd_per_1m_output_tokens,
            max_tokens=args.max_tokens,
        )

    run_dir = args.run_dir if args.run_dir is not None else (
        DEFAULT_RUNS_ROOT / args.manifest.stem
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    responses_path = run_dir / "responses.jsonl"
    responses, prior_spent_usd = _read_responses(responses_path)

    guard = BudgetGuard(max_spend_usd=args.max_spend, initial_spent_usd=prior_spent_usd)
    with responses_path.open("a") as out:
        for pair in manifest.pairs:
            if pair.pair_id in responses:
                continue  # resume: answered in a previous run
            guard.precheck(client.estimate_cost_usd(pair))
            resp = client.ask(pair)
            guard.record(resp.cost_usd)
            out.write(
                json.dumps(
                    {
                        "pair_id": resp.pair_id,
                        "text": resp.text,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                    }
                )
                + "\n"
            )
            out.flush()
            responses[resp.pair_id] = resp.text

    report = score_responses(manifest, responses)
    (run_dir / "report.json").write_text(render_report(report))
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "manifest": str(args.manifest),
                "mode": "recorded" if args.recorded is not None else "tinker",
                "spent_usd": guard.spent_usd,
                "max_spend_usd": args.max_spend,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"verdict: {report['verdict']}")
    for reason in report["verdict_reasons"]:
        print(f"  - {reason}")
    print(f"report: {run_dir / 'report.json'}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
