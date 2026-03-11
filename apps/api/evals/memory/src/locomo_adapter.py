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
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def _retry_on_rate_limit(fn, max_retries: int = 3, base_delay: float = 2.0):
    """Retry a callable on 429/503 with exponential backoff.

    Handles both requests.HTTPError (API calls) and Groq SDK exceptions (groq.RateLimitError).
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            # Extract status code from either requests or Groq SDK exceptions
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            if status == 0:
                status = getattr(exc, "status_code", 0)
            if status in (429, 503) and attempt < max_retries:
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
                delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
                print(f"    Rate limited ({status}), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise


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

QA_SYSTEM_PROMPT = """\
You are answering questions about a person based on extracted memory facts.

Steps:
1. Identify which facts are relevant to the question
2. If the question requires combining multiple facts, trace the reasoning chain
3. If the question asks about time ("when", "how long ago"), use the dates in the facts and the current date provided
4. Answer concisely -- typically 1-5 words for factual questions, 1-2 sentences for open-ended

If the facts don't contain enough information, say "I don't know".
Do NOT guess or infer beyond what the facts state."""


def _group_facts_for_context(facts: list[dict], current_date: str = "") -> str:
    """Group accumulated facts by category for structured QA context."""
    groups: dict[str, list[str]] = {}
    for fact in facts:
        cat = fact.get("category", "general")
        text = fact.get("fact_text", "")
        if not text:
            continue
        heading = {
            "identity": "Identity",
            "background": "Background",
            "goals": "Goals & Plans",
            "preferences": "Preferences & Habits",
            "repertoire": "Repertoire",
            "events": "Events & Milestones",
        }.get(cat, "Other")
        groups.setdefault(heading, []).append(text)

    if not groups:
        return "(No facts extracted)"

    lines = []
    if current_date:
        lines.append(f"Current date: {current_date}")
        lines.append("")
    for heading in ["Identity", "Background", "Goals & Plans", "Preferences & Habits",
                     "Repertoire", "Events & Milestones", "Other"]:
        if heading in groups:
            lines.append(f"## {heading}")
            for text in groups[heading]:
                lines.append(f"- {text}")
            lines.append("")

    return "\n".join(lines)


def _filter_relevant_facts(
    question: str,
    facts: list[dict],
    max_facts: int = 20,
) -> list[dict]:
    """Filter facts to those most relevant to the question.

    Two-stage: keyword pre-filter, then cosine similarity re-rank if needed.
    """
    if len(facts) <= max_facts:
        return facts

    # Stage 1: keyword pre-filter
    q_lower = question.lower()
    q_words = set(re.sub(r"[^\w\s]", "", q_lower).split())
    stopwords = {"a", "an", "the", "is", "was", "were", "are", "do", "does", "did",
                 "what", "when", "where", "who", "how", "which", "that", "this",
                 "to", "of", "in", "for", "on", "with", "at", "by", "from", "and",
                 "or", "not", "but", "if", "about", "has", "had", "have", "be", "been"}
    q_keywords = q_words - stopwords

    scored: list[tuple[int, dict]] = []
    for fact in facts:
        fact_lower = fact.get("fact_text", "").lower()
        hits = sum(1 for kw in q_keywords if kw in fact_lower)
        scored.append((hits, fact))

    keyword_matches = [fact for hits, fact in scored if hits > 0]

    if len(keyword_matches) <= max_facts:
        if len(keyword_matches) < max_facts:
            remaining = [fact for hits, fact in scored if hits == 0]
            keyword_matches.extend(remaining[: max_facts - len(keyword_matches)])
        return keyword_matches

    # Stage 2: cosine similarity re-rank
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([question])
    fact_texts = [f.get("fact_text", "") for f in keyword_matches]
    fact_embs = model.encode(fact_texts)
    sims = util.cos_sim(q_emb, fact_embs)[0]

    ranked = sorted(
        zip(keyword_matches, sims.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [fact for fact, _ in ranked[:max_facts]]


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

    def _do_call():
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

    return _retry_on_rate_limit(_do_call)


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
    """Answer a question given memory context, with rate-limit retry."""
    def _do_call():
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": QA_SYSTEM_PROMPT,
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        return response.choices[0].message.content or ""

    try:
        return _retry_on_rate_limit(_do_call)
    except Exception as exc:
        print(f"    QA call failed after retries: {exc}")
        return ""


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
# Date parsing helper
# ---------------------------------------------------------------------------

def _parse_locomo_date(date_str: str) -> str:
    """Parse LoCoMo date format '1:56 pm on 8 May, 2023' to 'YYYY-MM-DD'."""
    from datetime import datetime
    try:
        date_part = date_str.split(" on ", 1)[1] if " on " in date_str else date_str
        dt = datetime.strptime(date_part.strip(), "%d %B, %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return "2024-01-01"


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
        conv_id = conv.get("sample_id") or conv.get("conversation_id", "unknown")
        result = LoCoMoResult(conversation_id=conv_id)

        # Phase 1: Extract facts from dialogue turns
        accumulated_facts: list[dict] = []
        turn_idx = 0

        conversation = conv.get("conversation", {})
        speaker_a = conversation.get("speaker_a", "User")

        # Iterate over sessions in order (session_1, session_2, ...)
        session_num = 1
        while True:
            session_key = f"session_{session_num}"
            session_turns = conversation.get(session_key)
            if session_turns is None:
                break
            if not isinstance(session_turns, list):
                session_num += 1
                continue

            # Get date for this session
            date_key = f"session_{session_num}_date_time"
            date_str = conversation.get(date_key, "")
            session_date = _parse_locomo_date(date_str) if date_str else "2024-01-01"

            # Pair consecutive turns: speaker_a = user, speaker_b = assistant
            i = 0
            while i < len(session_turns) - 1:
                turn_a = session_turns[i]
                turn_b = session_turns[i + 1]

                if turn_a.get("speaker") == speaker_a:
                    user_msg = turn_a.get("text", "")
                    assistant_msg = turn_b.get("text", "")
                else:
                    user_msg = turn_b.get("text", "")
                    assistant_msg = turn_a.get("text", "")

                today = session_date

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
                    i += 2
                    continue

                # Process ADD operations
                for fact in extract_result.get("add", []):
                    fact_id = f"fact_{conv_id}_{turn_idx}_{len(accumulated_facts)}"
                    accumulated_facts.append({
                        "id": fact_id,
                        "fact_text": fact.get("fact_text", ""),
                        "category": fact.get("category", ""),
                        "permanent": fact.get("permanent", True),
                        "entities": fact.get("entities", []),
                        "relations": fact.get("relations", []),
                    })

                # Process UPDATE operations
                for fact in extract_result.get("update", []):
                    old_id = fact.get("existing_fact_id", "")
                    accumulated_facts = [
                        f for f in accumulated_facts if f.get("id") != old_id
                    ]
                    fact_id = f"fact_{conv_id}_{turn_idx}_{len(accumulated_facts)}"
                    fact_text = fact.get("new_fact_text", "") or fact.get("fact_text", "")
                    accumulated_facts.append({
                        "id": fact_id,
                        "fact_text": fact_text,
                        "category": fact.get("category", ""),
                        "permanent": fact.get("permanent", True),
                        "entities": fact.get("entities", []),
                        "relations": fact.get("relations", []),
                    })

                turn_idx += 1
                i += 2

            session_num += 1

        result.extraction_count = len(accumulated_facts)

        # Phase 2: Answer QA pairs using accumulated facts as context
        # Determine latest date from dialogue for temporal reasoning
        last_date = "2024-01-01"
        conversation = conv.get("conversation", {})
        for sn in range(1, 100):
            dk = f"session_{sn}_date_time"
            ds = conversation.get(dk, "")
            if ds:
                parsed = _parse_locomo_date(ds)
                if parsed > last_date:
                    last_date = parsed

        qa_pairs = conv.get("qa", [])
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
                # Filter facts relevant to this question
                relevant_facts = _filter_relevant_facts(question, accumulated_facts)
                context = _group_facts_for_context(relevant_facts, current_date=last_date)

                prediction = _answer_question(question, context, groq_client)
                entry = {
                    "key": qa_cache_key,
                    "conversation_id": conv_id,
                    "qa_idx": qa_idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "category": category,
                    "prediction": prediction,
                    "n_facts_total": len(accumulated_facts),
                    "n_facts_filtered": len(relevant_facts),
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
