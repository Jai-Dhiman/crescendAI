"""Full-session visible eval orchestrator (issue #70).

Runs a complete local session end-to-end with a headed (or headless) browser and
scripted chat turns that verify memory recall. Prints a per-criterion PASS/FAIL
report.

Usage:
    cd apps/evals
    uv run python e2e_full_session.py [--headless] [--conversation-id ID]
        [--no-seed] [--reply-timeout MS] [--recording PATH] [--piece-slug SLUG]

Key flags:
    --headless              Run browser headlessly (default: False — visible window)
    --conversation-id ID    Skip drive_persisted(); use existing conversation
    --no-seed               Skip canary fact seeding
    --max-chunks N          Max WebM chunks (default: 6)
    --timeout SECS          Per-event WS timeout (default: 120)
    --reply-timeout MS      Per-turn Playwright hard deadline ms (default: 90000)
    --screenshot-dir PATH   Dir for per-phase screenshots (default: /tmp/e2e-full)
    --wrangler-url URL      API URL (default: http://localhost:8787)
    --web-url URL           Web URL (default: http://localhost:3000)
    --db-dsn DSN            Postgres DSN for canary seeding
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Report types (offline-testable — no live imports at module level)
# ---------------------------------------------------------------------------

ToolOutcome = Literal["TOOL_RENDERED", "TEXT_ONLY", "UNKNOWN"]


@dataclass
class FullSessionReport:
    """Per-criterion outcomes for the full session eval run."""
    conversation_id: str
    # (a) V6 artifact rendered (synthesis-message in DOM)
    criteria_a: bool
    # (b) Headline match + components
    criteria_b_headline: bool
    criteria_b_components: bool
    # (c) Exercise confirm flow (None = no prescription)
    criteria_c: bool | None
    # (e) Memory recall — canary tokens found in chat reply
    criteria_e_recall: bool
    criteria_e_tokens_found: list[str]
    criteria_e_tokens_missing: list[str]
    # (f) Tool action outcome (non-fatal)
    criteria_f_tool_outcome: ToolOutcome
    errors: list[str] = field(default_factory=list)

    @property
    def overall(self) -> bool:
        """PASS iff (a), (b-headline), (b-components), and (e) all pass; (f) is non-fatal."""
        if self.errors:
            return False
        return (
            self.criteria_a
            and self.criteria_b_headline
            and self.criteria_b_components
            and self.criteria_e_recall
        )


def format_report(report: FullSessionReport) -> str:
    """Render a human-readable per-criterion report string."""
    def _yn(v: bool | None, skip_label: str = "SKIP") -> str:
        if v is None:
            return skip_label
        return "PASS" if v else "FAIL"

    lines = [
        "",
        "=" * 65,
        "FULL SESSION EVAL REPORT",
        "=" * 65,
        f"Conversation: {report.conversation_id}",
        "",
        f"(a) V6 artifact rendered (synthesis in DOM):     {_yn(report.criteria_a)}",
        f"(b) Headline match (DOM == WS text):             {_yn(report.criteria_b_headline)}",
        f"(b) Components rendered (cards present):         {_yn(report.criteria_b_components)}",
    ]

    if report.criteria_c is None:
        lines.append("(c) Exercise confirm flow:                       SKIP (no prescription)")
    else:
        lines.append(f"(c) Exercise confirm flow:                       {_yn(report.criteria_c)}")

    recall_detail = ""
    if report.criteria_e_tokens_found:
        recall_detail = f" found={report.criteria_e_tokens_found}"
    if report.criteria_e_tokens_missing:
        recall_detail += f" missing={report.criteria_e_tokens_missing}"
    lines.append(
        f"(e) Memory recall — canary tokens in reply:      {_yn(report.criteria_e_recall)}{recall_detail}"
    )
    lines.append(
        f"(f) Tool action (ExerciseSetCard):               {report.criteria_f_tool_outcome} (non-fatal)"
    )

    if report.errors:
        lines.append("")
        lines.append("Errors:")
        for err in report.errors:
            lines.append(f"  - {err}")

    lines.append("")
    lines.append(f"OVERALL: {'PASS' if report.overall else 'FAIL'}")
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat turn scripts
# ---------------------------------------------------------------------------

RECALL_TURN = "What do you know about me and what am I currently preparing?"
TOOL_TURN = "Give me a left-hand drill for bars 1-4."

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
DEFAULT_API_DIR = REPO_ROOT / "apps" / "api"
DEFAULT_DB_DSN = "postgresql://jdhiman:postgres@localhost:5432/crescendai_dev"
DEFAULT_SCREENSHOT_DIR = Path("/tmp/e2e-full")


def run(
    recording: Path = DEFAULT_RECORDING,
    piece_slug: str = DEFAULT_PIECE_SLUG,
    wrangler_url: str = "http://localhost:8787",
    web_url: str = "http://localhost:3000",
    api_dir: Path | None = None,
    db_dsn: str = DEFAULT_DB_DSN,
    screenshot_dir: Path = DEFAULT_SCREENSHOT_DIR,
    max_chunks: int = 6,
    timeout_per_event: float = 120.0,
    headless: bool = False,
    slow_mo: int = 700,
    conversation_id: str | None = None,
    skip_seed: bool = False,
    reply_timeout_ms: int = 90000,
) -> int:
    """Run the full visible session eval. Returns 0 on PASS, 1 on FAIL.

    All Playwright waits are explicitly bounded. reply_timeout_ms defaults to
    90s to absorb Workers AI glm cold-start latency. Raise --reply-timeout on
    a cold stack (challenge caution: cold-start can exceed 90s).
    """
    from memory_seeder import (
        CANARY_TOKENS,
        CanarySeed,
        get_debug_student_id,
        seed_canary_facts,
    )
    from shared.local_session import check_services, drive_persisted
    from ui_verifier import (
        _auth_browser,
        _component_testid,
        _save_screenshot,
        run_chat_turns,
    )
    from playwright.sync_api import sync_playwright

    effective_api_dir = api_dir if api_dir is not None else DEFAULT_API_DIR
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # --- Pre-flight ---
    print(f"[full-eval] Health check: {wrangler_url}")
    try:
        check_services(wrangler_url)
    except RuntimeError as exc:
        print(f"ERROR: services not ready: {exc}", file=sys.stderr)
        print("Run `just dev` and `just seed-fingerprint` first.", file=sys.stderr)
        return 1

    # --- Step 1: Seed canary facts ---
    canary: CanarySeed | None = None
    if not skip_seed:
        print("[full-eval] Fetching debug student_id from API...")
        try:
            student_id = get_debug_student_id(wrangler_url)
        except RuntimeError as exc:
            print(f"ERROR: cannot get debug student_id: {exc}", file=sys.stderr)
            return 1

        print(f"[full-eval] Seeding canary facts for student_id={student_id}")
        try:
            canary = seed_canary_facts(student_id=student_id, db_dsn=db_dsn)
        except RuntimeError as exc:
            print(f"ERROR: canary seed failed: {exc}", file=sys.stderr)
            return 1
        assert canary is not None  # always set in this branch; narrows CanarySeed | None
        print(f"[full-eval] Seeded {canary.rows_inserted} canary rows: {canary.tokens}")
    else:
        print("[full-eval] Skipping canary seed (--no-seed).")

    # --- Step 2: Drive recording (or skip if --conversation-id provided) ---
    if conversation_id is None:
        if not recording.exists():
            print(f"ERROR: recording not found: {recording}", file=sys.stderr)
            return 1

        print(f"[full-eval] Driving recording: {recording.name} (piece={piece_slug})")
        try:
            cap = drive_persisted(
                recording=recording,
                piece_slug=piece_slug,
                wrangler_url=wrangler_url,
                api_dir=effective_api_dir,
                timeout_per_event=timeout_per_event,
                max_chunks=max_chunks,
            )
        except RuntimeError as exc:
            print(f"ERROR: drive_persisted() failed: {exc}", file=sys.stderr)
            return 1

        conversation_id = cap.conversation_id
        expected_headline = cap.headline_text
        component_types = [c.get("type", "") for c in cap.components]
        has_prescription = cap.prescribed_exercise is not None
        is_fallback = cap.is_fallback

        print(f"[full-eval] Session: {cap.session_id}")
        print(f"[full-eval] Conversation: {conversation_id}")
        print(f"[full-eval] Headline: {expected_headline!r}")
        print(f"[full-eval] is_fallback: {is_fallback}")

        if is_fallback:
            print("ERROR: synthesis is_fallback=true — V6 artifact not produced.", file=sys.stderr)
            return 1
    else:
        print(f"[full-eval] Using existing conversation_id: {conversation_id}")
        expected_headline = ""
        component_types = []
        has_prescription = False

    # --- Step 3: Browser assertions + scripted chat ---
    errors: list[str] = []
    criteria_a = False
    criteria_b_headline = False
    criteria_b_components = False
    criteria_c: bool | None = None
    criteria_e_recall = False
    tokens_found: list[str] = []
    tokens_missing: list[str] = []
    criteria_f_outcome: ToolOutcome = "UNKNOWN"

    nav_timeout_ms = 15000
    synthesis_timeout_ms = 30000
    screenshot_paths: list[str] = []

    print(f"[full-eval] Opening {'headed' if not headless else 'headless'} browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=slow_mo if not headless else 0)
        context = browser.new_context()
        page = context.new_page()

        try:
            # Authenticate
            try:
                _auth_browser(page, wrangler_url)
            except RuntimeError as exc:
                errors.append(f"Auth failed: {exc}")
                _save_screenshot(page, screenshot_dir / "auth-fail.png")
                screenshot_paths.append(str(screenshot_dir / "auth-fail.png"))
                _print_and_exit(conversation_id, criteria_a, criteria_b_headline,
                                criteria_b_components, criteria_c, criteria_e_recall,
                                tokens_found, tokens_missing, criteria_f_outcome,
                                errors, screenshot_paths)
                return 1

            # Navigate to conversation
            conv_url = f"{web_url}/app/c/{conversation_id}"
            print(f"[full-eval] Navigating to: {conv_url}")
            try:
                page.goto(conv_url, timeout=nav_timeout_ms)
            except Exception as exc:
                errors.append(f"Navigation failed: {exc}")
                return 1

            # Wait for synthesis message (criterion a)
            if expected_headline:
                try:
                    page.wait_for_selector("[data-testid='synthesis-message']", timeout=synthesis_timeout_ms)
                    criteria_a = True
                except Exception as exc:
                    errors.append(f"synthesis-message not found within {synthesis_timeout_ms}ms: {exc}")

                if criteria_a:
                    hl_el = page.query_selector("[data-testid='synthesis-headline']")
                    if hl_el:
                        dom_text = (hl_el.inner_text() or "").strip()
                        criteria_b_headline = dom_text == expected_headline.strip()
                        if not criteria_b_headline:
                            errors.append(
                                f"Headline mismatch.\n  Expected: {expected_headline!r}\n  Got: {dom_text!r}"
                            )
                    else:
                        errors.append("synthesis-headline not found")

                    criteria_b_components = True
                    renderable = [t for t in component_types if t not in ("pending_exercise", "search_catalog_result")]
                    for ctype in renderable:
                        testid = _component_testid(ctype)
                        if testid and not page.query_selector(f"[data-testid='{testid}']"):
                            errors.append(f"Component card not found: type={ctype} testid={testid}")
                            criteria_b_components = False

                    if has_prescription:
                        confirm_btn = page.query_selector("[data-testid='confirm-exercise-button']")
                        if not confirm_btn:
                            errors.append("confirm-exercise-button not found (prescription expected)")
                            criteria_c = False
                        else:
                            try:
                                confirm_btn.click()
                                page.wait_for_selector("[data-testid='exercise-set-card']", timeout=10000)
                                criteria_c = True
                            except Exception as exc:
                                errors.append(f"Confirm flow failed: {exc}")
                                criteria_c = False
            else:
                # conversation_id provided directly — skip synthesis assertions
                criteria_a = True
                criteria_b_headline = True
                criteria_b_components = True

            # Screenshot after synthesis phase
            synth_shot = screenshot_dir / "01-synthesis.png"
            _save_screenshot(page, synth_shot)
            screenshot_paths.append(str(synth_shot))

            # --- Scripted chat turns ---
            chat_replies: list[str] = []
            try:
                chat_replies = [r.reply_text for r in run_chat_turns(
                    page=page,
                    turns=[RECALL_TURN, TOOL_TURN],
                    reply_timeout_ms=reply_timeout_ms,
                )]
            except RuntimeError as exc:
                errors.append(f"Chat turn failed: {exc}")

            # Screenshot after chat phase
            chat_shot = screenshot_dir / "02-chat.png"
            _save_screenshot(page, chat_shot)
            screenshot_paths.append(str(chat_shot))

            # Criterion (e): memory recall — canary tokens in recall reply
            if chat_replies:
                recall_reply = chat_replies[0]
                for token in CANARY_TOKENS:
                    if token in recall_reply:
                        tokens_found.append(token)
                    else:
                        tokens_missing.append(token)
                criteria_e_recall = len(tokens_found) > 0
            else:
                tokens_missing = list(CANARY_TOKENS)
                criteria_e_recall = False
                errors.append("No chat replies received — memory recall cannot be assessed")

            # Criterion (f): tool action — non-fatal
            if len(chat_replies) >= 2:
                tool_shot = screenshot_dir / "03-tool.png"
                _save_screenshot(page, tool_shot)
                screenshot_paths.append(str(tool_shot))

                # Check whether an exercise-set-card appeared
                exercise_card = page.query_selector("[data-testid='exercise-set-card']")
                if exercise_card:
                    criteria_f_outcome = "TOOL_RENDERED"
                else:
                    criteria_f_outcome = "TEXT_ONLY"
                print(f"[full-eval] Tool action result: {criteria_f_outcome} (non-fatal)")
            else:
                criteria_f_outcome = "UNKNOWN"

        except Exception as exc:
            errors.append(f"Unexpected error: {exc}")
        finally:
            context.close()
            browser.close()

    report = FullSessionReport(
        conversation_id=conversation_id,
        criteria_a=criteria_a,
        criteria_b_headline=criteria_b_headline,
        criteria_b_components=criteria_b_components,
        criteria_c=criteria_c,
        criteria_e_recall=criteria_e_recall,
        criteria_e_tokens_found=tokens_found,
        criteria_e_tokens_missing=tokens_missing,
        criteria_f_tool_outcome=criteria_f_outcome,
        errors=errors,
    )

    print(format_report(report))
    if screenshot_paths:
        print(f"Screenshots: {screenshot_paths}")

    return 0 if report.overall else 1


def _print_and_exit(
    conversation_id: str,
    criteria_a: bool,
    criteria_b_headline: bool,
    criteria_b_components: bool,
    criteria_c: bool | None,
    criteria_e_recall: bool,
    tokens_found: list[str],
    tokens_missing: list[str],
    criteria_f_outcome: ToolOutcome,
    errors: list[str],
    screenshot_paths: list[str],
) -> None:
    report = FullSessionReport(
        conversation_id=conversation_id,
        criteria_a=criteria_a,
        criteria_b_headline=criteria_b_headline,
        criteria_b_components=criteria_b_components,
        criteria_c=criteria_c,
        criteria_e_recall=criteria_e_recall,
        criteria_e_tokens_found=tokens_found,
        criteria_e_tokens_missing=tokens_missing,
        criteria_f_tool_outcome=criteria_f_outcome,
        errors=errors,
    )
    print(format_report(report))
    if screenshot_paths:
        print(f"Screenshots: {screenshot_paths}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="CrescendAI full session visible eval")
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--piece-slug", default=DEFAULT_PIECE_SLUG)
    parser.add_argument("--wrangler-url", default="http://localhost:8787")
    parser.add_argument("--web-url", default="http://localhost:3000")
    parser.add_argument("--api-dir", type=Path, default=None)
    parser.add_argument("--db-dsn", default=DEFAULT_DB_DSN)
    parser.add_argument("--screenshot-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-event WS timeout (s)")
    parser.add_argument("--reply-timeout", type=int, default=90000, help="Per-turn browser deadline (ms)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--slow-mo", type=int, default=700)
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--no-seed", action="store_true")
    args = parser.parse_args()

    sys.exit(run(
        recording=args.recording,
        piece_slug=args.piece_slug,
        wrangler_url=args.wrangler_url,
        web_url=args.web_url,
        api_dir=args.api_dir,
        db_dsn=args.db_dsn,
        screenshot_dir=args.screenshot_dir,
        max_chunks=args.max_chunks,
        timeout_per_event=args.timeout,
        reply_timeout_ms=args.reply_timeout,
        headless=args.headless,
        slow_mo=args.slow_mo,
        conversation_id=args.conversation_id,
        skip_seed=args.no_seed,
    ))


if __name__ == "__main__":
    _cli()
