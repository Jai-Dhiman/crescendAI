"""LoCoMo benchmark adapter for CrescendAI memory evaluation.

LoCoMo (ACL 2024) tests long-conversation memory via QA pairs over
multi-session dialogues. We adapt it to test our chat extraction pipeline:
feed dialogue turns through extract-chat, then answer QA pairs using
accumulated facts as context.

Two modes: offline (cached JSONL) and live (API + Groq, populates cache).
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"
_DEV_VARS_PATH = Path(__file__).parents[3] / ".dev.vars"

API_BASE = "http://localhost:8787"

CATEGORY_NAMES = {
    1: "Single-hop",
    2: "Multi-hop",
    3: "Temporal",
    4: "Open-ended",
    5: "Adversarial",
}


# ---------------------------------------------------------------------------
# Token-level F1 (official LoCoMo metric)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, remove articles, punctuation, extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens) if pred_tokens else 0.0
    recall = num_common / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_auth_token() -> str:
    import requests

    resp = requests.post(f"{API_BASE}/api/auth/debug")
    resp.raise_for_status()
    return resp.json()["token"]


def _call_extract_chat(
    token: str,
    user_message: str,
    assistant_response: str,
    existing_facts: list[dict],
    today: str,
) -> dict:
    import requests

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
# Groq QA answering
# ---------------------------------------------------------------------------

def _load_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    if _DEV_VARS_PATH.exists():
        for line in _DEV_VARS_PATH.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GROQ_API_KEY not found in env or apps/api/.dev.vars")


def _answer_question(question: str, context: str, groq_client) -> str:
    """Answer a question given memory context."""
    system = (
        "Answer the question concisely based on the provided context. "
        "If the context doesn't contain enough information, say 'I don't know'."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["key"]] = entry
    return cache


def _save_cache_entry(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoCoMoResult:
    conversation_id: str
    total_qa: int = 0
    overall_f1: float = 0.0
    per_category_f1: dict = field(default_factory=dict)  # category_id -> f1
    extraction_count: int = 0  # total facts extracted
    qa_results: list[dict] = field(default_factory=list)  # per-QA detail


# ---------------------------------------------------------------------------
# Main assessment flow
# ---------------------------------------------------------------------------

def run_locomo_assessment(
    data_path: Path,
    live: bool = False,
    max_samples: int = 2,
    extraction_cache_path: Path | None = None,
    qa_cache_path: Path | None = None,
) -> list[LoCoMoResult]:
    """Run LoCoMo benchmark assessment.

    Args:
        data_path: Path to locomo10.json.
        live: If True, call APIs. If False, use cached responses.
        max_samples: Max conversations to process.
        extraction_cache_path: Path to extraction cache JSONL.
        qa_cache_path: Path to QA cache JSONL.

    Returns:
        List of LoCoMoResult, one per conversation.
    """
    if extraction_cache_path is None:
        extraction_cache_path = DATA_DIR / "locomo_extraction_cache.jsonl"
    if qa_cache_path is None:
        qa_cache_path = DATA_DIR / "locomo_qa_cache.jsonl"

    with open(data_path) as f:
        conversations = json.load(f)

    extraction_cache = _load_cache(extraction_cache_path)
    qa_cache = _load_cache(qa_cache_path)

    # Get auth token and groq client if live
    token = None
    groq_client = None
    if live:
        token = _get_auth_token()
        import groq
        groq_client = groq.Groq(api_key=_load_groq_key())

    results: list[LoCoMoResult] = []

    for conv in conversations[:max_samples]:
        conv_id = conv["conversation_id"]
        result = LoCoMoResult(conversation_id=conv_id)

        # Phase 1: Extract facts from dialogue turns
        accumulated_facts: list[dict] = []
        turn_idx = 0

        for session in conv.get("dialogue", []):
            turns = session.get("turns", [])
            # Pair consecutive turns as (user_msg, assistant_msg)
            for i in range(0, len(turns) - 1, 2):
                user_turn = turns[i]
                assistant_turn = turns[i + 1] if i + 1 < len(turns) else None

                if assistant_turn is None:
                    continue

                user_msg = user_turn.get("text", "")
                assistant_msg = assistant_turn.get("text", "")

                # Parse timestamp to YYYY-MM-DD
                timestamp = user_turn.get("timestamp", "")
                today = timestamp[:10] if len(timestamp) >= 10 else "2024-01-01"

                cache_key = f"{conv_id}_turn_{turn_idx}"

                if cache_key in extraction_cache:
                    extract_result = extraction_cache[cache_key].get("result", {})
                elif live:
                    assert token is not None
                    extract_result = _call_extract_chat(
                        token, user_msg, assistant_msg, accumulated_facts, today
                    )
                    entry = {
                        "key": cache_key,
                        "conversation_id": conv_id,
                        "turn_idx": turn_idx,
                        "user_message": user_msg,
                        "assistant_response": assistant_msg,
                        "result": extract_result,
                    }
                    extraction_cache[cache_key] = entry
                    _save_cache_entry(extraction_cache_path, entry)
                else:
                    turn_idx += 1
                    continue

                # Process ADD/UPDATE results to maintain accumulated_facts
                actions = extract_result.get("actions", [])
                for action in actions:
                    op = action.get("operation", "add")
                    if op == "add":
                        accumulated_facts.append({
                            "fact_text": action.get("fact_text", ""),
                            "category": action.get("category", ""),
                            "permanent": action.get("permanent", True),
                        })
                    elif op == "update":
                        # Find and replace the superseded fact
                        supersedes = action.get("supersedes_fact_id")
                        if supersedes is not None:
                            # Remove by index if it's an integer reference
                            try:
                                idx_to_remove = int(supersedes)
                                if 0 <= idx_to_remove < len(accumulated_facts):
                                    accumulated_facts.pop(idx_to_remove)
                            except (ValueError, TypeError):
                                pass
                        accumulated_facts.append({
                            "fact_text": action.get("fact_text", ""),
                            "category": action.get("category", ""),
                            "permanent": action.get("permanent", True),
                        })

                turn_idx += 1

        result.extraction_count = len(accumulated_facts)

        # Phase 2: Answer QA pairs using accumulated facts as context
        context_lines = []
        for fact in accumulated_facts:
            cat = fact.get("category", "general")
            text = fact.get("fact_text", "")
            context_lines.append(f"- [{cat}] {text}")
        context = "\n".join(context_lines) if context_lines else "(No facts extracted)"

        qa_pairs = conv.get("qa_pairs", [])
        category_scores: dict[int, list[float]] = {}
        all_scores: list[float] = []

        for qa_idx, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            gold_answer = qa.get("answer", "")
            category = qa.get("category", 0)

            qa_cache_key = f"{conv_id}_qa_{qa_idx}"

            if qa_cache_key in qa_cache:
                prediction = qa_cache[qa_cache_key].get("prediction", "")
            elif live:
                if groq_client is None:
                    import groq
                    groq_client = groq.Groq(api_key=_load_groq_key())
                prediction = _answer_question(question, context, groq_client)
                entry = {
                    "key": qa_cache_key,
                    "conversation_id": conv_id,
                    "qa_idx": qa_idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "category": category,
                    "prediction": prediction,
                }
                qa_cache[qa_cache_key] = entry
                _save_cache_entry(qa_cache_path, entry)
            else:
                continue

            f1 = token_f1(prediction, gold_answer)
            all_scores.append(f1)

            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(f1)

            result.qa_results.append({
                "qa_idx": qa_idx,
                "question": question,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "category": category,
                "f1": f1,
            })

        result.total_qa = len(all_scores)
        result.overall_f1 = sum(all_scores) / len(all_scores) if all_scores else 0.0

        for cat_id, scores in category_scores.items():
            result.per_category_f1[cat_id] = sum(scores) / len(scores) if scores else 0.0

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def print_results(results: list[LoCoMoResult]) -> None:
    print("\n=== LoCoMo Benchmark Assessment ===\n")

    if not results:
        print("  No results. Run with --live to populate cache.")
        return

    for r in results:
        print(f"  Conversation: {r.conversation_id}")
        print(f"    Facts extracted: {r.extraction_count}")
        print(f"    QA pairs: {r.total_qa}")
        print(f"    Overall F1: {r.overall_f1:.3f}")
        if r.per_category_f1:
            for cat_id in sorted(r.per_category_f1.keys()):
                cat_name = CATEGORY_NAMES.get(cat_id, f"Category {cat_id}")
                print(f"    {cat_name}: {r.per_category_f1[cat_id]:.3f}")
        print()

    # Aggregate across all conversations
    all_f1 = []
    agg_category: dict[int, list[float]] = {}
    total_facts = 0

    for r in results:
        if r.total_qa > 0:
            all_f1.append(r.overall_f1)
            total_facts += r.extraction_count
            for cat_id, f1 in r.per_category_f1.items():
                if cat_id not in agg_category:
                    agg_category[cat_id] = []
                agg_category[cat_id].append(f1)

    if all_f1:
        print(f"--- Aggregate (n={len(results)} conversations) ---")
        print(f"  Overall F1:       {sum(all_f1) / len(all_f1):.3f}")
        print(f"  Total facts:      {total_facts}")
        for cat_id in sorted(agg_category.keys()):
            cat_name = CATEGORY_NAMES.get(cat_id, f"Category {cat_id}")
            scores = agg_category[cat_id]
            print(f"  {cat_name:<16} {sum(scores) / len(scores):.3f}")
    else:
        print("  No QA results to aggregate. Run with --live to populate cache.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    live = "--live" in sys.argv

    max_samples = 2  # default
    if "--locomo-samples" in sys.argv:
        idx = sys.argv.index("--locomo-samples")
        max_samples = int(sys.argv[idx + 1])

    data_path = DATA_DIR / "locomo10.json"
    if not data_path.exists():
        print(f"LoCoMo data not found at {data_path}")
        print("Download from: https://github.com/snap-research/locomo")
        return

    results = run_locomo_assessment(data_path, live=live, max_samples=max_samples)
    print_results(results)


if __name__ == "__main__":
    main()
