"""Chat extraction evaluation: test live API endpoint against expected facts.

Layer 5 of the memory eval system. Loads chat extraction scenarios, calls the
live Rust API at localhost:8787 for each exchange, and evaluates extracted facts
against gold-standard expected facts.

Two modes: offline (cached JSONL) and live (API calls, populates cache).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date  # noqa: F401 - used in _check_temporal_accuracy
from pathlib import Path

import requests

from .scenarios import ChatExtractionScenario, ExpectedChatFact, load_chat_scenarios

DATA_DIR = Path(__file__).parents[1] / "data"

API_BASE = "http://localhost:8787"

# Lazy-loaded sentence transformer for cosine similarity fallback
_SENTENCE_MODEL = None


def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SENTENCE_MODEL


def _strip_regex(pattern: str) -> str:
    """Convert a regex pattern to plain text for embedding comparison."""
    text = pattern
    text = text.replace("(?i)", "")
    text = re.sub(r"\(([^)]+)\)", lambda m: " ".join(m.group(1).split("|")), text)
    text = text.replace(".*", " ")
    text = text.replace("-?", "-")
    text = re.sub(r"[\\^$*+?\[\]{}|]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _get_auth_token() -> str:
    """Get a debug auth token from the local dev server."""
    resp = requests.post(f"{API_BASE}/api/auth/debug")
    resp.raise_for_status()
    data = resp.json()
    return data["token"]


def _call_extract_chat(
    token: str,
    user_message: str,
    assistant_response: str,
    existing_facts: list[dict],
    today: str,
) -> dict:
    """Call the live API extraction endpoint."""
    resp = requests.post(
        f"{API_BASE}/api/memory/extract-chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "user_message": user_message,
            "assistant_response": assistant_response,
            "existing_facts": existing_facts,
            "today": today,
        },
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["key"]] = entry
    return cache


def _save_cache_entry(cache_path: Path, entry: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatExtractionResult:
    scenario_id: str
    extraction_recall: float = 0.0
    extraction_precision: float = 0.0
    category_accuracy: float = 0.0
    operation_accuracy: float = 0.0
    temporal_accuracy: float = 0.0
    selectivity_pass: bool = True
    e2e_context_hit: bool = True
    raw_outputs: list[dict] = field(default_factory=list)
    matched_expected: list[str] = field(default_factory=list)
    unmatched_expected: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fact matching (2-pass: regex then cosine similarity)
# ---------------------------------------------------------------------------

def _match_chat_facts(
    extracted_facts: list[dict],
    expected_facts: list[ExpectedChatFact],
) -> dict[int, tuple[str, bool, bool]]:
    """Match extracted facts against expected using regex then cosine similarity.

    Returns dict mapping extracted_index -> (expected_id, category_correct, operation_correct).
    Uses greedy 1:1 assignment to prevent double-matching.
    """
    results: dict[int, tuple[str, bool, bool]] = {}
    matched_ef_ids: set[str] = set()
    unmatched_indices: list[int] = []

    # Pass 1: regex matching
    for ei, ef_extracted in enumerate(extracted_facts):
        fact_text = ef_extracted.get("fact_text", "")
        found = False
        for ef in expected_facts:
            if ef.id in matched_ef_ids:
                continue
            try:
                if re.search(ef.fact_text_pattern, fact_text, re.IGNORECASE):
                    cat_ok = ef.category == ef_extracted.get("category", "")
                    op_ok = ef.operation == ef_extracted.get("operation", "")
                    results[ei] = (ef.id, cat_ok, op_ok)
                    matched_ef_ids.add(ef.id)
                    found = True
                    break
            except re.error:
                continue
        if not found:
            unmatched_indices.append(ei)

    # Pass 2: cosine similarity for remaining unmatched
    remaining_efs = [ef for ef in expected_facts if ef.id not in matched_ef_ids]
    if unmatched_indices and remaining_efs:
        model = _get_sentence_model()
        extracted_texts = [extracted_facts[i].get("fact_text", "") for i in unmatched_indices]
        ref_texts = [
            ef.gold_fact_text if ef.gold_fact_text else _strip_regex(ef.fact_text_pattern)
            for ef in remaining_efs
        ]

        extracted_embeddings = model.encode(extracted_texts)
        ref_embeddings = model.encode(ref_texts)

        from sentence_transformers import util
        sim_matrix = util.cos_sim(extracted_embeddings, ref_embeddings)

        # Greedy best-first 1:1 assignment
        used_extracted: set[int] = set()
        used_ef: set[int] = set()

        pairs = []
        for ei_local in range(len(unmatched_indices)):
            for ef_local in range(len(remaining_efs)):
                pairs.append((float(sim_matrix[ei_local][ef_local]), ei_local, ef_local))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for score, ei_local, ef_local in pairs:
            if score < 0.55:
                break
            if ei_local in used_extracted or ef_local in used_ef:
                continue
            ei_global = unmatched_indices[ei_local]
            ef = remaining_efs[ef_local]
            ef_extracted = extracted_facts[ei_global]
            cat_ok = ef.category == ef_extracted.get("category", "")
            op_ok = ef.operation == ef_extracted.get("operation", "")
            results[ei_global] = (ef.id, cat_ok, op_ok)
            used_extracted.add(ei_local)
            used_ef.add(ef_local)

    return results


# ---------------------------------------------------------------------------
# Temporal accuracy check
# ---------------------------------------------------------------------------

def _check_temporal_accuracy(
    extracted_facts: list[dict],
    expected_facts: list[ExpectedChatFact],
    match_map: dict[int, tuple[str, bool, bool]],
) -> float:
    """Check if extracted invalid_at dates are within 3 days of expected."""
    temporal_checks = 0
    temporal_correct = 0

    expected_by_id = {ef.id: ef for ef in expected_facts}

    for ei, (ef_id, _, _) in match_map.items():
        ef = expected_by_id.get(ef_id)
        if ef is None or ef.invalid_at is None:
            continue

        temporal_checks += 1
        extracted_invalid_at = extracted_facts[ei].get("invalid_at")
        if extracted_invalid_at is None:
            continue

        try:
            expected_date = date.fromisoformat(ef.invalid_at)
            extracted_date = date.fromisoformat(extracted_invalid_at)
            if abs((extracted_date - expected_date).days) <= 3:
                temporal_correct += 1
        except ValueError:
            continue

    if temporal_checks == 0:
        return 1.0
    return temporal_correct / temporal_checks


# ---------------------------------------------------------------------------
# Pre-existing facts -> API format
# ---------------------------------------------------------------------------

def _pre_existing_to_api_format(facts: list[ExpectedChatFact]) -> list[dict]:
    """Convert pre-existing ExpectedChatFact objects to API-compatible dict format."""
    result = []
    for f in facts:
        result.append({
            "id": f.id,
            "fact_text": f.gold_fact_text if f.gold_fact_text else _strip_regex(f.fact_text_pattern),
            "category": f.category,
        })
    return result


# ---------------------------------------------------------------------------
# Main evaluation flow
# ---------------------------------------------------------------------------

def run_chat_extraction_assessment(
    scenarios: list[ChatExtractionScenario],
    live: bool = False,
    cache_path: Path | None = None,
) -> list[ChatExtractionResult]:
    """Run chat extraction assessment.

    Args:
        scenarios: Chat extraction scenarios to test.
        live: If True, call the live API. If False, use cached responses.
        cache_path: Path to cache JSONL file.
    """
    if cache_path is None:
        cache_path = DATA_DIR / "chat_extraction_cache.jsonl"

    cache = _load_cache(cache_path)

    # Obtain auth token once if running live
    token: str | None = None
    if live:
        try:
            token = _get_auth_token()
        except requests.ConnectionError:
            print(
                "ERROR: Cannot connect to local dev server at "
                f"{API_BASE}. Start it with: cd apps/api && wrangler dev"
            )
            return []
        except requests.HTTPError as exc:
            print(f"ERROR: Failed to get debug auth token: {exc}")
            return []

    results: list[ChatExtractionResult] = []

    for scenario in scenarios:
        # Track accumulated facts, starting with pre-existing facts
        accumulated_facts = _pre_existing_to_api_format(scenario.pre_existing_facts)

        all_extracted: list[dict] = []
        raw_outputs: list[dict] = []
        skipped = False

        for exchange in scenario.exchanges:
            cache_key = f"{scenario.id}_{exchange.id}"

            if cache_key in cache:
                api_response = cache[cache_key]["response"]
            elif live:
                assert token is not None
                try:
                    api_response = _call_extract_chat(
                        token=token,
                        user_message=exchange.user_message,
                        assistant_response=exchange.assistant_response,
                        existing_facts=accumulated_facts,
                        today=exchange.session_date,
                    )
                except requests.ConnectionError:
                    print(
                        f"  ERROR: Connection lost during {scenario.id}/{exchange.id}. "
                        f"Is the dev server still running?"
                    )
                    skipped = True
                    break
                except requests.HTTPError as exc:
                    print(f"  ERROR: API error for {scenario.id}/{exchange.id}: {exc}")
                    skipped = True
                    break

                # Cache the response
                entry = {
                    "key": cache_key,
                    "scenario_id": scenario.id,
                    "exchange_id": exchange.id,
                    "user_message": exchange.user_message,
                    "assistant_response": exchange.assistant_response,
                    "existing_facts": accumulated_facts,
                    "today": exchange.session_date,
                    "response": api_response,
                }
                cache[cache_key] = entry
                _save_cache_entry(cache_path, entry)
            else:
                # Not cached and not live -- skip scenario
                results.append(ChatExtractionResult(
                    scenario_id=scenario.id,
                    raw_outputs=[{"status": "[NOT CACHED]"}],
                ))
                skipped = True
                break

            raw_outputs.append(api_response)

            # Process ADD operations
            for fact in api_response.get("add", []):
                fact_id = str(uuid.uuid4())
                new_fact = {
                    "id": fact_id,
                    "fact_text": fact.get("fact_text", ""),
                    "category": fact.get("category", ""),
                }
                accumulated_facts.append(new_fact)
                all_extracted.append({
                    "fact_text": fact.get("fact_text", ""),
                    "category": fact.get("category", ""),
                    "operation": "add",
                    "invalid_at": fact.get("invalid_at"),
                    "permanent": fact.get("permanent", True),
                })

            # Process UPDATE operations
            for fact in api_response.get("update", []):
                old_id = fact.get("existing_fact_id", "")
                # Remove old fact from accumulated list
                accumulated_facts = [
                    f for f in accumulated_facts if f.get("id") != old_id
                ]
                # Add new version
                fact_id = str(uuid.uuid4())
                fact_text = fact.get("new_fact_text", "") or fact.get("fact_text", "")
                new_fact = {
                    "id": fact_id,
                    "fact_text": fact_text,
                    "category": fact.get("category", ""),
                }
                accumulated_facts.append(new_fact)
                all_extracted.append({
                    "fact_text": fact_text,
                    "category": fact.get("category", ""),
                    "operation": "update",
                    "invalid_at": fact.get("invalid_at"),
                    "permanent": fact.get("permanent", True),
                    "old_fact_id": old_id,
                })

        if skipped:
            continue

        # --- Evaluate against expected facts ---

        expected = scenario.expected_facts
        match_map = _match_chat_facts(all_extracted, expected)

        # Matched / unmatched / hallucinated
        matched_ids: list[str] = []
        unmatched_expected_ids: list[str] = [ef.id for ef in expected]
        hallucinated: list[str] = []

        category_correct = 0
        operation_correct = 0
        total_matched = 0

        for ei, ef_extracted in enumerate(all_extracted):
            if ei in match_map:
                ef_id, cat_ok, op_ok = match_map[ei]
                matched_ids.append(ef_id)
                if ef_id in unmatched_expected_ids:
                    unmatched_expected_ids.remove(ef_id)
                total_matched += 1
                if cat_ok:
                    category_correct += 1
                if op_ok:
                    operation_correct += 1
            else:
                hallucinated.append(ef_extracted.get("fact_text", ""))

        n_expected = len(expected)
        n_extracted = len(all_extracted)

        extraction_recall = len(matched_ids) / n_expected if n_expected > 0 else 1.0
        extraction_precision = total_matched / n_extracted if n_extracted > 0 else 1.0
        category_accuracy = category_correct / total_matched if total_matched > 0 else 1.0
        operation_accuracy = operation_correct / total_matched if total_matched > 0 else 1.0

        # Temporal accuracy
        temporal_accuracy = _check_temporal_accuracy(all_extracted, expected, match_map)

        # Selectivity check
        selectivity_pass = True
        if scenario.category == "selectivity":
            if not expected:
                # Selectivity scenario expects zero extractions
                selectivity_pass = len(all_extracted) == 0
            else:
                selectivity_pass = extraction_recall >= 0.8

        # E2E context hit
        e2e_context_hit = True
        if scenario.category == "e2e":
            context_str = " ".join(f.get("fact_text", "") for f in accumulated_facts)
            for ef in expected:
                pattern = ef.fact_text_pattern
                try:
                    if not re.search(pattern, context_str, re.IGNORECASE):
                        # Fallback: check gold_fact_text as substring
                        if ef.gold_fact_text and ef.gold_fact_text.lower() not in context_str.lower():
                            e2e_context_hit = False
                            break
                except re.error:
                    if ef.gold_fact_text and ef.gold_fact_text.lower() not in context_str.lower():
                        e2e_context_hit = False
                        break

        results.append(ChatExtractionResult(
            scenario_id=scenario.id,
            extraction_recall=extraction_recall,
            extraction_precision=extraction_precision,
            category_accuracy=category_accuracy,
            operation_accuracy=operation_accuracy,
            temporal_accuracy=temporal_accuracy,
            selectivity_pass=selectivity_pass,
            e2e_context_hit=e2e_context_hit,
            raw_outputs=raw_outputs,
            matched_expected=matched_ids,
            unmatched_expected=unmatched_expected_ids,
            hallucinated=hallucinated,
        ))

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(
    results: list[ChatExtractionResult],
    scenarios: list[ChatExtractionScenario] | None = None,
) -> None:
    print("\n=== Chat Extraction Assessment (Layer 5) ===\n")

    # Build scenario category lookup
    scenario_categories: dict[str, str] = {}
    if scenarios:
        scenario_categories = {s.id: s.category for s in scenarios}

    all_recall: list[float] = []
    all_precision: list[float] = []
    all_category_acc: list[float] = []
    all_operation_acc: list[float] = []
    temporal_accs: list[float] = []
    selectivity_total = 0
    selectivity_passed = 0
    e2e_total = 0
    e2e_hit = 0
    skipped = 0

    for r in results:
        if r.raw_outputs and r.raw_outputs[0].get("status") == "[NOT CACHED]":
            print(f"  [SKIP] {r.scenario_id} (not cached, run with --live)")
            skipped += 1
            continue

        # Determine pass/fail
        is_pass = r.extraction_recall >= 0.8 and r.selectivity_pass and r.e2e_context_hit
        status = "PASS" if is_pass else "FAIL"

        print(f"  [{status}] {r.scenario_id}")
        print(
            f"    recall={r.extraction_recall:.2f} prec={r.extraction_precision:.2f} "
            f"cat_acc={r.category_accuracy:.2f} op_acc={r.operation_accuracy:.2f} "
            f"temp_acc={r.temporal_accuracy:.2f}"
        )
        if r.unmatched_expected:
            print(f"    Missing expected: {r.unmatched_expected}")
        if r.hallucinated:
            for h in r.hallucinated[:3]:
                print(f"    Hallucinated: {h[:80]}...")

        all_recall.append(r.extraction_recall)
        all_precision.append(r.extraction_precision)
        all_category_acc.append(r.category_accuracy)
        all_operation_acc.append(r.operation_accuracy)
        temporal_accs.append(r.temporal_accuracy)

        cat = scenario_categories.get(r.scenario_id, "")
        if cat == "selectivity":
            selectivity_total += 1
            if r.selectivity_pass:
                selectivity_passed += 1
        if cat == "e2e":
            e2e_total += 1
            if r.e2e_context_hit:
                e2e_hit += 1

    n = len(all_recall)
    if n == 0:
        print("\n  No results to aggregate. Run with --live to populate cache.")
        return

    print(f"\n--- Aggregate (n={n}, skipped={skipped}) ---")
    print(f"  Extraction recall:    {sum(all_recall)/n:.3f}")
    print(f"  Extraction precision: {sum(all_precision)/n:.3f}")
    print(f"  Category accuracy:    {sum(all_category_acc)/n:.3f}")
    print(f"  Operation accuracy:   {sum(all_operation_acc)/n:.3f}")
    if temporal_accs:
        # Filter to non-vacuous temporal scenarios (accuracy < 1.0 or has temporal expected)
        print(f"  Temporal accuracy:    {sum(temporal_accs)/len(temporal_accs):.3f}")
    if selectivity_total > 0:
        print(f"  Selectivity rate:     {selectivity_passed}/{selectivity_total} ({selectivity_passed/selectivity_total:.2f})")
    else:
        print(f"  Selectivity rate:     N/A (no selectivity scenarios)")
    if e2e_total > 0:
        print(f"  E2E context hit rate: {e2e_hit}/{e2e_total} ({e2e_hit/e2e_total:.2f})")
    else:
        print(f"  E2E context hit rate: N/A (no e2e scenarios)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    live = "--live" in sys.argv
    scenarios_path = DATA_DIR / "chat_scenarios.jsonl"

    if not scenarios_path.exists():
        print(
            "Chat scenarios not found at "
            f"{scenarios_path}.\n"
            "Run: uv run python -m src.build_chat_scenarios"
        )
        return

    scenarios = load_chat_scenarios(scenarios_path)

    if "--scenario" in sys.argv:
        idx = sys.argv.index("--scenario")
        sid = sys.argv[idx + 1]
        scenarios = [s for s in scenarios if s.id == sid]

    results = run_chat_extraction_assessment(scenarios, live=live)
    print_results(results, scenarios=scenarios)


if __name__ == "__main__":
    main()
