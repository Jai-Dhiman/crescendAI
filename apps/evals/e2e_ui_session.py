"""End-to-end UI session test orchestrator (issue #68).

Drives ONE real recording through the live local pipeline:
  1. drive_persisted(): real MuQ+AMT -> V6 synthesis on glm-4.7-flash@WorkersAI
     -> persisted to debug user's conversation in local DB
  2. verify_ui(): Playwright navigates to /app/c/<conversationId>, asserts that
     the synthesis renders correctly (headline, components, optional confirm flow)

Usage:
    cd apps/evals
    uv run python -m e2e_ui_session [--recording <wav>] [--piece-slug <slug>]

Options:
    --recording PATH      WAV file to drive (default: nocturne_op9no2)
    --piece-slug SLUG     Piece slug for set_piece WS message
    --wrangler-url URL    API URL (default: http://localhost:8787)
    --web-url URL         Web URL (default: http://localhost:3000)
    --screenshot PATH     Where to save the Playwright screenshot
    --max-chunks N        Maximum WebM chunks to send (default: 6)
    --timeout SECS        Per-event WS timeout in seconds (default: 120)
    --no-headless         Run Playwright in a visible browser window
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shared.local_session import check_services, drive_persisted
from ui_verifier import verify_ui

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDING = (
    REPO_ROOT
    / "model"
    / "data"
    / "evals"
    / "practice_eval"
    / "nocturne_op9no2"
    / "audio"
    / "_aySCutsVVQ.wav"
)
DEFAULT_PIECE_SLUG = "nocturne_op9no2"
DEFAULT_SCREENSHOT = Path("/tmp/e2e-ui-session.png")


def _lowest_dim(chunk_scores: list[list[float]]) -> str | None:
    """Return the name of the dimension with the lowest mean score across all chunks.

    Dimension order matches the 6-dim output of MuQ:
      0=dynamics, 1=timing, 2=pedaling, 3=articulation, 4=phrasing, 5=interpretation
    """
    dim_names = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]
    if not chunk_scores:
        return None
    n_dims = min(len(dim_names), min(len(row) for row in chunk_scores))
    if n_dims == 0:
        return None
    means = [
        sum(row[i] for row in chunk_scores) / len(chunk_scores)
        for i in range(n_dims)
    ]
    return dim_names[means.index(min(means))]


def run(
    recording: Path,
    piece_slug: str,
    wrangler_url: str = "http://localhost:8787",
    web_url: str = "http://localhost:3000",
    screenshot_path: Path = DEFAULT_SCREENSHOT,
    max_chunks: int = 6,
    timeout_per_event: float = 120.0,
    headless: bool = True,
) -> int:
    """Run the full e2e test. Returns 0 on pass, 1 on failure."""
    # --- Pre-flight ---
    if not recording.exists():
        print(f"ERROR: recording not found: {recording}", file=sys.stderr)
        print(
            "Populate it with: model/data/evals/practice_eval/nocturne_op9no2/audio/_aySCutsVVQ.wav",
            file=sys.stderr,
        )
        return 1

    print(f"[e2e] Health check: {wrangler_url}")
    try:
        check_services(wrangler_url)
    except RuntimeError as exc:
        print(f"ERROR: services not ready: {exc}", file=sys.stderr)
        print("Run `just dev` (or `just dev-muq`) and `just seed-fingerprint` first.", file=sys.stderr)
        return 1

    # --- Step 1: Drive the recording ---
    print(f"[e2e] Driving recording: {recording.name} (piece={piece_slug})")
    print(f"[e2e] Max chunks: {max_chunks}, timeout/event: {timeout_per_event}s")
    try:
        cap = drive_persisted(
            recording=recording,
            piece_slug=piece_slug,
            wrangler_url=wrangler_url,
            timeout_per_event=timeout_per_event,
            max_chunks=max_chunks,
        )
    except RuntimeError as exc:
        print(f"ERROR: drive_persisted() failed: {exc}", file=sys.stderr)
        return 1

    print(f"[e2e] Session: {cap.session_id}")
    print(f"[e2e] Conversation: {cap.conversation_id}")
    print(f"[e2e] Headline: {cap.headline_text!r}")
    print(f"[e2e] is_fallback: {cap.is_fallback}")
    print(f"[e2e] Components: {[c.get('type') for c in cap.components]}")
    print(f"[e2e] Prescribed exercise: {cap.prescribed_exercise is not None}")
    print(f"[e2e] Chunk scores rows: {len(cap.chunk_scores)}")

    if cap.is_fallback:
        print("ERROR: synthesis is_fallback=true — V6 artifact not produced.", file=sys.stderr)
        print("Check DO logs for v6 phase_error or v6 validation_error.", file=sys.stderr)
        return 1

    # Derive lowest dimension
    lowest_dim = _lowest_dim(cap.chunk_scores)
    print(f"[e2e] Lowest dimension: {lowest_dim}")

    # Component types present in synthesis
    component_types = [c.get("type", "") for c in cap.components]

    # --- Step 2: Verify in the web UI ---
    print(f"[e2e] Navigating to: {web_url}/app/c/{cap.conversation_id}")
    result = verify_ui(
        conversation_id=cap.conversation_id,
        expected_headline=cap.headline_text,
        expected_component_types=component_types,
        lowest_dim_name=lowest_dim,
        has_prescription=cap.prescribed_exercise is not None,
        web_url=web_url,
        api_url=wrangler_url,
        screenshot_path=screenshot_path,
        headless=headless,
    )

    # --- Report ---
    print()
    print("=" * 60)
    print("E2E UI SESSION RESULT")
    print("=" * 60)
    print(f"(a) V6 artifact (isFallback=false, synthesis rendered): {'PASS' if result.criteria_a_v6_artifact else 'FAIL'}")
    print(f"(b) Headline match (DOM == WS text):                    {'PASS' if result.criteria_b_headline_match else 'FAIL'}")
    print(f"(b) Components rendered (cards present):                {'PASS' if result.criteria_b_components_rendered else 'FAIL'}")
    if result.criteria_c_confirm_flow is not None:
        print(f"(c) Confirm->assign->ExerciseSetCard reveal:            {'PASS' if result.criteria_c_confirm_flow else 'FAIL'}")
    else:
        print("(c) Confirm flow:                                       SKIP (no prescription)")
    if result.criteria_d_dimension_in_headline is not None:
        print(f"(d) Lowest dim '{lowest_dim}' in headline:              {'PASS' if result.criteria_d_dimension_in_headline else 'FAIL'}")
    else:
        print("(d) Dimension in headline:                              SKIP (no chunk scores or no dim)")
    print()
    if result.screenshot_path:
        print(f"Screenshot: {result.screenshot_path}")
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")
    print()
    overall = "PASS" if result.passed else "FAIL"
    print(f"OVERALL: {overall}")
    print("=" * 60)

    return 0 if result.passed else 1


def _cli() -> None:
    parser = argparse.ArgumentParser(description="CrescendAI e2e UI session test")
    parser.add_argument(
        "--recording",
        type=Path,
        default=DEFAULT_RECORDING,
        help="WAV file to drive through the pipeline",
    )
    parser.add_argument(
        "--piece-slug",
        default=DEFAULT_PIECE_SLUG,
        help="Piece slug for set_piece WS message",
    )
    parser.add_argument("--wrangler-url", default="http://localhost:8787")
    parser.add_argument("--web-url", default="http://localhost:3000")
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=DEFAULT_SCREENSHOT,
        help="Path to save the Playwright screenshot",
    )
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-event WS timeout (seconds)")
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    sys.exit(
        run(
            recording=args.recording,
            piece_slug=args.piece_slug,
            wrangler_url=args.wrangler_url,
            web_url=args.web_url,
            screenshot_path=args.screenshot,
            max_chunks=args.max_chunks,
            timeout_per_event=args.timeout,
            headless=not args.no_headless,
        )
    )


if __name__ == "__main__":
    _cli()
