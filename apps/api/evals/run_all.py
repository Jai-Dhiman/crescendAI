from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

from shared.inference_cache import find_cache_dir


CACHE_ROOT = Path(__file__).parent / "data" / "inference_cache"
DEV_VARS_PATH = Path(__file__).parent.parent / ".dev.vars"
WORKER_BASE = "http://localhost:8787"


def preflight() -> bool:
    """Run preflight checks. Returns True if all pass."""
    checks: list[tuple[str, bool, str]] = []

    # 1. Inference cache exists
    try:
        cache_dir = find_cache_dir(CACHE_ROOT)
        checks.append(("Inference cache", True, str(cache_dir)))
    except FileNotFoundError as e:
        checks.append(("Inference cache", False, str(e)))

    # 2. ANTHROPIC_API_KEY env var
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    checks.append((
        "ANTHROPIC_API_KEY env",
        len(anthropic_key) > 0,
        "set" if anthropic_key else "not set",
    ))

    # 3. GROQ_API_KEY env var
    groq_key = os.environ.get("GROQ_API_KEY", "")
    checks.append((
        "GROQ_API_KEY env",
        len(groq_key) > 0,
        "set" if groq_key else "not set",
    ))

    # 4. wrangler dev responding
    try:
        resp = requests.get(f"{WORKER_BASE}/health", timeout=3)
        checks.append((
            "Worker /health",
            resp.status_code == 200,
            f"HTTP {resp.status_code}",
        ))
    except requests.ConnectionError:
        checks.append(("Worker /health", False, "connection refused"))
    except requests.Timeout:
        checks.append(("Worker /health", False, "timeout"))

    # 5. D1 seeded (exercises endpoint)
    try:
        resp = requests.get(f"{WORKER_BASE}/api/exercises", timeout=5)
        has_data = resp.status_code == 200 and len(resp.json()) > 0
        checks.append((
            "D1 seeded (exercises)",
            has_data,
            f"HTTP {resp.status_code}, {len(resp.json()) if resp.status_code == 200 else 0} exercises",
        ))
    except (requests.ConnectionError, requests.Timeout):
        checks.append(("D1 seeded (exercises)", False, "worker not reachable"))
    except Exception as e:
        checks.append(("D1 seeded (exercises)", False, str(e)))

    # 6. Worker LLM keys in .dev.vars
    if DEV_VARS_PATH.exists():
        dev_vars_content = DEV_VARS_PATH.read_text()
        has_groq = "GROQ_API_KEY" in dev_vars_content
        has_anthropic = "ANTHROPIC_API_KEY" in dev_vars_content
        both = has_groq and has_anthropic
        missing = []
        if not has_groq:
            missing.append("GROQ_API_KEY")
        if not has_anthropic:
            missing.append("ANTHROPIC_API_KEY")
        detail = "both present" if both else f"missing: {', '.join(missing)}"
        checks.append(("Worker .dev.vars keys", both, detail))
    else:
        checks.append(("Worker .dev.vars keys", False, f"{DEV_VARS_PATH} not found"))

    # Print results
    print()
    print(f"  {'Check':<25} {'Status':>8}  Detail")
    print(f"  {'-' * 60}")
    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<25} {status:>8}  {detail}")
        if not passed:
            all_passed = False
    print()

    return all_passed


def run_observation_quality() -> None:
    """Run the observation quality eval suite."""
    from observation_quality.eval_observation_quality import main as obs_main

    obs_main(
        cache_dir=CACHE_ROOT,
        traces_dir=Path(__file__).parent.parent.parent.parent / "data" / "eval" / "traces",
        reports_dir=Path(__file__).parent / "reports",
        wrangler_url=WORKER_BASE,
    )


def run_subagent_reasoning() -> None:
    """Run the subagent reasoning eval suite."""
    from subagent_reasoning.eval_subagent_reasoning import main as sub_main

    sub_main(reports_dir=Path(__file__).parent / "reports")


def main() -> None:
    parser = argparse.ArgumentParser(description="CrescendAI Pipeline Eval Runner")
    parser.add_argument(
        "--suite",
        choices=["obs", "subagent"],
        help="Run a specific eval suite (default: run all)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks",
    )
    args = parser.parse_args()

    if not args.skip_preflight:
        print("Running preflight checks...")
        if not preflight():
            print("Preflight checks failed. Use --skip-preflight to override.")
            sys.exit(1)
        print("All preflight checks passed.")

    if args.suite == "obs":
        run_observation_quality()
    elif args.suite == "subagent":
        run_subagent_reasoning()
    else:
        run_observation_quality()
        run_subagent_reasoning()


if __name__ == "__main__":
    main()
