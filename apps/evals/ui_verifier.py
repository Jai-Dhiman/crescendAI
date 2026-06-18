"""Playwright-based UI verifier for the e2e session test (issue #68).

Authenticates the browser as the debug user, navigates to the conversation
that drive_persisted() created, and asserts that the V6 synthesis is rendered
correctly in the web UI.

Usage (standalone, for debugging):
    cd apps/evals
    uv run python -m ui_verifier --conversation-id <conv_id> \\
        --headline "Your phrasing..." \\
        --web-url http://localhost:3000 \\
        --screenshot-path /tmp/e2e-screenshot.png

API:
    from ui_verifier import verify_ui, VerificationResult
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright


@dataclass
class VerificationResult:
    """Result of verifying a conversation in the web UI."""
    conversation_id: str
    # Success criteria (a)-(d) from issue #68
    criteria_a_v6_artifact: bool  # synthesis-message present in DOM (isFallback=false proven by server emit)
    criteria_b_headline_match: bool  # DOM headline text == expected_headline_text
    criteria_b_components_rendered: bool  # all expected component types have a card in DOM
    criteria_c_confirm_flow: bool | None  # None if no prescription; True if confirm succeeded
    criteria_d_dimension_in_headline: bool | None  # None if no dimension; True if found in headline
    screenshot_path: Path | None
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.errors:
            return False
        if not (self.criteria_a_v6_artifact and self.criteria_b_headline_match and self.criteria_b_components_rendered):
            return False
        return True


def _auth_browser(page: Page, api_url: str) -> None:
    """Authenticate the browser context as the debug user via /api/auth/debug."""
    resp = page.request.post(f"{api_url}/api/auth/debug")
    if resp.status != 200:
        raise RuntimeError(
            f"Browser debug auth failed: {resp.status} {resp.text()}"
        )
    # The debug endpoint sets an HttpOnly cookie — the browser context now holds it.


def verify_ui(
    conversation_id: str,
    expected_headline: str,
    expected_component_types: list[str],
    lowest_dim_name: str | None,
    has_prescription: bool,
    web_url: str = "http://localhost:3000",
    api_url: str = "http://localhost:8787",
    screenshot_path: Path | None = None,
    headless: bool = True,
    nav_timeout_ms: int = 15000,
    synthesis_timeout_ms: int = 20000,
) -> VerificationResult:
    """Open the conversation in a browser and assert the synthesis renders correctly.

    Args:
        conversation_id: The conversationId from drive_persisted().
        expected_headline: The headline text that the synthesis WS event emitted.
        expected_component_types: Component types from the synthesis WS event
            (e.g. ["pending_exercise"]) that should appear as rendered cards.
        lowest_dim_name: The name of the lowest-mean-score dimension, or None.
        has_prescription: True if drive_persisted() returned a prescribed_exercise.
        web_url: Base URL for the web app.
        api_url: Base URL for the API (for debug auth).
        screenshot_path: If set, save a screenshot here after assertion.
        headless: Whether to run Playwright headlessly.
        nav_timeout_ms: Timeout for page navigation.
        synthesis_timeout_ms: Timeout waiting for synthesis-message to appear.

    Returns:
        VerificationResult with per-criterion outcomes.
    """
    errors: list[str] = []

    with sync_playwright() as p:
        browser: Browser = p.chromium.launch(headless=headless)
        context: BrowserContext = browser.new_context()
        page: Page = context.new_page()

        try:
            # Step 1: Authenticate the browser context
            try:
                _auth_browser(page, api_url)
            except RuntimeError as exc:
                errors.append(f"Auth failed: {exc}")
                return VerificationResult(
                    conversation_id=conversation_id,
                    criteria_a_v6_artifact=False,
                    criteria_b_headline_match=False,
                    criteria_b_components_rendered=False,
                    criteria_c_confirm_flow=None,
                    criteria_d_dimension_in_headline=None,
                    screenshot_path=None,
                    errors=errors,
                )

            # Step 2: Navigate to the conversation
            conv_url = f"{web_url}/app/c/{conversation_id}"
            try:
                page.goto(conv_url, timeout=nav_timeout_ms)
            except Exception as exc:
                errors.append(f"Navigation failed: {exc}")
                return VerificationResult(
                    conversation_id=conversation_id,
                    criteria_a_v6_artifact=False,
                    criteria_b_headline_match=False,
                    criteria_b_components_rendered=False,
                    criteria_c_confirm_flow=None,
                    criteria_d_dimension_in_headline=None,
                    screenshot_path=None,
                    errors=errors,
                )

            # Step 3: Wait for synthesis-message to appear in the DOM
            try:
                page.wait_for_selector("[data-testid='synthesis-message']", timeout=synthesis_timeout_ms)
            except Exception as exc:
                errors.append(f"synthesis-message not found within {synthesis_timeout_ms}ms: {exc}")
                _save_screenshot(page, screenshot_path)
                return VerificationResult(
                    conversation_id=conversation_id,
                    criteria_a_v6_artifact=False,
                    criteria_b_headline_match=False,
                    criteria_b_components_rendered=False,
                    criteria_c_confirm_flow=None,
                    criteria_d_dimension_in_headline=None,
                    screenshot_path=screenshot_path,
                    errors=errors,
                )

            # Criterion (a): synthesis-message present => V6 artifact rendered
            criteria_a = True

            # Criterion (b-1): headline text matches
            headline_el = page.query_selector("[data-testid='synthesis-headline']")
            if not headline_el:
                errors.append("synthesis-headline element not found")
                criteria_b_headline = False
            else:
                dom_headline = (headline_el.inner_text() or "").strip()
                criteria_b_headline = dom_headline == expected_headline.strip()
                if not criteria_b_headline:
                    errors.append(
                        f"Headline mismatch.\n  Expected: {expected_headline!r}\n  Got: {dom_headline!r}"
                    )

            # Criterion (b-2): each expected renderable component has a card
            renderable_types = [t for t in expected_component_types if t != "pending_exercise"]
            criteria_b_components = True
            for ctype in renderable_types:
                testid = _component_testid(ctype)
                if testid and not page.query_selector(f"[data-testid='{testid}']"):
                    errors.append(f"Component card not found: type={ctype} testid={testid}")
                    criteria_b_components = False

            # Criterion (c): confirm -> assign -> ExerciseSetCard reveal
            criteria_c: bool | None = None
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

            # Criterion (d): lowest dim name appears in headline text
            criteria_d: bool | None = None
            if lowest_dim_name and criteria_b_headline:
                headline_el2 = page.query_selector("[data-testid='synthesis-headline']")
                if headline_el2:
                    text = (headline_el2.inner_text() or "").lower()
                    criteria_d = lowest_dim_name.lower() in text

            # Screenshot
            saved_path = _save_screenshot(page, screenshot_path)

        except Exception as exc:
            errors.append(f"Unexpected error: {exc}")
            saved_path = _save_screenshot(page, screenshot_path)
            return VerificationResult(
                conversation_id=conversation_id,
                criteria_a_v6_artifact=False,
                criteria_b_headline_match=False,
                criteria_b_components_rendered=False,
                criteria_c_confirm_flow=None,
                criteria_d_dimension_in_headline=None,
                screenshot_path=saved_path,
                errors=errors,
            )
        finally:
            context.close()
            browser.close()

    return VerificationResult(
        conversation_id=conversation_id,
        criteria_a_v6_artifact=criteria_a,
        criteria_b_headline_match=criteria_b_headline,
        criteria_b_components_rendered=criteria_b_components,
        criteria_c_confirm_flow=criteria_c,
        criteria_d_dimension_in_headline=criteria_d,
        screenshot_path=saved_path,
        errors=errors,
    )


def _component_testid(component_type: str) -> str | None:
    """Map a component type to its data-testid. Returns None if no testid exists."""
    mapping = {
        "exercise_set": "exercise-set-card",
    }
    return mapping.get(component_type)


def _save_screenshot(page: Page, path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path))
        return path
    except Exception as exc:
        print(f"[warn] screenshot save failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI entrypoint (for operator use / debugging)
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a conversation's synthesis rendering in the web UI"
    )
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--headline", required=True, help="Expected headline text")
    parser.add_argument(
        "--component-types",
        default="",
        help="Comma-separated expected component types (e.g. pending_exercise)",
    )
    parser.add_argument("--lowest-dim", default=None, help="Lowest dimension name")
    parser.add_argument("--has-prescription", action="store_true")
    parser.add_argument("--web-url", default="http://localhost:3000")
    parser.add_argument("--api-url", default="http://localhost:8787")
    parser.add_argument("--screenshot", default=None, help="Path to save screenshot")
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    component_types = [t.strip() for t in args.component_types.split(",") if t.strip()]
    screenshot_path = Path(args.screenshot) if args.screenshot else None

    result = verify_ui(
        conversation_id=args.conversation_id,
        expected_headline=args.headline,
        expected_component_types=component_types,
        lowest_dim_name=args.lowest_dim,
        has_prescription=args.has_prescription,
        web_url=args.web_url,
        api_url=args.api_url,
        screenshot_path=screenshot_path,
        headless=not args.no_headless,
    )

    print(json.dumps({
        "passed": result.passed,
        "criteria_a_v6_artifact": result.criteria_a_v6_artifact,
        "criteria_b_headline_match": result.criteria_b_headline_match,
        "criteria_b_components_rendered": result.criteria_b_components_rendered,
        "criteria_c_confirm_flow": result.criteria_c_confirm_flow,
        "criteria_d_dimension_in_headline": result.criteria_d_dimension_in_headline,
        "screenshot_path": str(result.screenshot_path) if result.screenshot_path else None,
        "errors": result.errors,
    }, indent=2))

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    _cli()
