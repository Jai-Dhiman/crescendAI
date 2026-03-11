# Memory System: LoCoMo Benchmark + Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a competitive LoCoMo F1 score (>65, beating Mem0's 58.44) by fixing the adapter, adding structured retrieval with CoT QA, and implementing lightweight entity-hop retrieval.

**Architecture:** Fix structural bugs in `locomo_adapter.py` (wrong field names, no retry, no fact IDs), then layer on relevance filtering + fact grouping + CoT QA prompt + entity extraction + entity-hop retrieval. Extend `CHAT_EXTRACTION_SYSTEM` prompt in `prompts.rs` with optional entity/relation output. All retrieval logic lives in the Python adapter; Rust API changes are minimal (backward-compatible prompt extension).

**Tech Stack:** Python harness (`apps/api/evals/memory/`), Rust/WASM API (`apps/api/src/`), Groq Llama 3.3 70B, sentence-transformers for relevance filtering.

**Spec:** `docs/superpowers/specs/2026-03-10-memory-locomo-design.md`

---

## Chunk 1: Infrastructure Fixes

### Task 1: Add retry wrapper for API and Groq rate limits

**Files:**
- Modify: `apps/api/evals/memory/src/locomo_adapter.py:62-94`

- [ ] **Step 1: Write `_retry_on_rate_limit` wrapper**

Add after the existing imports (line 18), before `DATA_DIR`:

```python
import time


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
```

- [ ] **Step 2: Wrap `_call_extract_chat` with retry**

Replace `_call_extract_chat` (lines 74-94):

```python
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
```

- [ ] **Step 3: Wrap `_answer_question` with retry**

Replace `_answer_question` (lines 112-128):

```python
def _answer_question(question: str, context: str, groq_client) -> str:
    """Answer a question given memory context, with rate-limit retry."""
    def _do_call():
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
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
```

(Note: `QA_SYSTEM_PROMPT` is defined in Task 4. For now use the existing inline string until Task 4 is implemented.)

- [ ] **Step 4: Verify module loads**

Run: `cd apps/api/evals/memory && uv run python -c "from src.locomo_adapter import _retry_on_rate_limit; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add apps/api/evals/memory/src/locomo_adapter.py
git commit -m "add retry wrapper for Groq rate limits in LoCoMo adapter"
```

---

### Task 2: Fix fact processing to match API response format

**Files:**
- Modify: `apps/api/evals/memory/src/locomo_adapter.py:257-284`

- [ ] **Step 1: Replace broken `actions` processing with correct `add`/`update` format**

Replace lines 257-284 (the `# Process ADD/UPDATE results` block through `turn_idx += 1`):

```python
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
```

Note: `entities` and `relations` fields are included from the start. They will be empty lists until Task 6 extends the extraction prompt to output them. This avoids needing to rewrite the same block in Task 7.

- [ ] **Step 2: Verify import consistency**

Run: `cd apps/api/evals/memory && uv run python -c "from src.locomo_adapter import run_locomo_assessment; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add apps/api/evals/memory/src/locomo_adapter.py
git commit -m "fix LoCoMo adapter to match API add/update response format"
```

---

### Task 3: Download LoCoMo data and smoke test

**Files:**
- Download: `apps/api/evals/memory/data/locomo10.json`
- Create: `apps/api/evals/memory/.gitignore` (add locomo data to gitignore)

- [ ] **Step 1: Find and download LoCoMo data**

The LoCoMo dataset is at the snap-research/LoCoMo GitHub repo. The file we need is the 10-conversation subset.

```bash
cd apps/api/evals/memory/data
# Clone just the data file from the LoCoMo repo
# The exact URL depends on the repo structure -- check:
# https://github.com/snap-research/LoCoMo
# Look for a JSON file containing conversations with dialogue + qa_pairs
```

If the repo structure doesn't have a clean `locomo10.json`, you may need to:
1. Clone the full repo to a temp dir
2. Copy the relevant data file
3. Rename to `locomo10.json`

Verify structure: the file should be a JSON array of objects with `conversation_id`, `dialogue` (array of sessions with `turns`), and `qa_pairs`.

- [ ] **Step 2: Add to gitignore**

```bash
echo "data/locomo10.json" >> apps/api/evals/memory/.gitignore
```

- [ ] **Step 3: Smoke test with 1 sample (offline -- just verify parsing)**

```bash
cd apps/api/evals/memory
uv run python -c "
import json
from pathlib import Path
data = json.loads(Path('data/locomo10.json').read_text())
print(f'Conversations: {len(data)}')
c = data[0]
print(f'Conv ID: {c[\"conversation_id\"]}')
sessions = c.get('dialogue', [])
total_turns = sum(len(s.get('turns', [])) for s in sessions)
print(f'Sessions: {len(sessions)}, Total turns: {total_turns}')
print(f'QA pairs: {len(c.get(\"qa_pairs\", []))}')
qa = c['qa_pairs'][0]
print(f'Sample QA: category={qa.get(\"category\")}, q={qa[\"question\"][:60]}...')
"
```

- [ ] **Step 4: Commit**

```bash
git add apps/api/evals/memory/.gitignore
git commit -m "add LoCoMo data to gitignore, download locomo10.json"
```

---

## Chunk 2: Structured Retrieval + QA (Approach B)

### Task 4: Add CoT QA system prompt and fact grouping

**Files:**
- Modify: `apps/api/evals/memory/src/locomo_adapter.py:112-128` (QA prompt), `289-294` (context building)

- [ ] **Step 1: Add CoT QA system prompt constant**

Add after `CATEGORY_NAMES` (line 32), before the token_f1 section:

```python
QA_SYSTEM_PROMPT = """\
You are answering questions about a person based on extracted memory facts.

Steps:
1. Identify which facts are relevant to the question
2. If the question requires combining multiple facts, trace the reasoning chain
3. If the question asks about time ("when", "how long ago"), use the dates in the facts and the current date provided
4. Answer concisely -- typically 1-5 words for factual questions, 1-2 sentences for open-ended

If the facts don't contain enough information, say "I don't know".
Do NOT guess or infer beyond what the facts state."""

```

- [ ] **Step 2: Add fact grouping function**

Add after `QA_SYSTEM_PROMPT`:

```python
def _group_facts_for_context(facts: list[dict], current_date: str = "") -> str:
    """Group accumulated facts by category for structured QA context."""
    groups: dict[str, list[str]] = {}
    for fact in facts:
        cat = fact.get("category", "general")
        text = fact.get("fact_text", "")
        if not text:
            continue
        # Map categories to readable headings
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
```

- [ ] **Step 3: Update `_answer_question` to use the CoT prompt**

Ensure `_answer_question` references `QA_SYSTEM_PROMPT` (should already be done from Task 1 Step 3).

- [ ] **Step 4: Update context building in `run_locomo_assessment`**

Replace lines 289-294 (the `# Phase 2: Answer QA pairs` context building):

```python
        # Phase 2: Answer QA pairs using accumulated facts as context
        # Determine latest date from dialogue for temporal reasoning
        last_date = "2024-01-01"
        for session in conv.get("dialogue", []):
            for turn in session.get("turns", []):
                ts = turn.get("timestamp", "")
                if len(ts) >= 10 and ts[:10] > last_date:
                    last_date = ts[:10]

        context = _group_facts_for_context(accumulated_facts, current_date=last_date)
```

- [ ] **Step 5: Verify module imports cleanly**

Run: `cd apps/api/evals/memory && uv run python -c "from src.locomo_adapter import _group_facts_for_context, QA_SYSTEM_PROMPT; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add apps/api/evals/memory/src/locomo_adapter.py
git commit -m "add CoT QA prompt and fact grouping for LoCoMo"
```

---

### Task 5: Add relevance filtering for large fact sets

**Files:**
- Modify: `apps/api/evals/memory/src/locomo_adapter.py`

- [ ] **Step 1: Add relevance filter function**

Add after `_group_facts_for_context`:

```python
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
    # Remove stopwords
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

    # Keep all facts with at least 1 keyword hit
    keyword_matches = [fact for hits, fact in scored if hits > 0]

    if len(keyword_matches) <= max_facts:
        # Pad with highest-scored non-matches if we have room
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
```

- [ ] **Step 2: Replace the entire Phase 2 block with per-question filtering**

In `run_locomo_assessment`, replace the entire Phase 2 block (from `# Phase 2: Answer QA pairs` through the end of the QA loop, approximately lines 288-342 in the original file). Keep only the `last_date` computation from Task 4 before this block, then replace everything from `qa_pairs = conv.get(...)` through the end of the QA result appending:

```python
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
```

- [ ] **Step 3: Verify import**

Run: `cd apps/api/evals/memory && uv run python -c "from src.locomo_adapter import _filter_relevant_facts; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add apps/api/evals/memory/src/locomo_adapter.py
git commit -m "add relevance filtering for LoCoMo QA context"
```

---

## Chunk 3: Knowledge Graph Layer (Approach C)

### Task 6: Extend extraction prompt for optional entity output

**Files:**
- Modify: `apps/api/src/services/prompts.rs:485-503` (CHAT_EXTRACTION_SYSTEM output format)

- [ ] **Step 1: Add optional entity fields to the output JSON schema in prompt**

In `CHAT_EXTRACTION_SYSTEM`, replace the `## Output` section. Find the text starting with `## Output` and ending with `- invalid_at: ISO date string (YYYY-MM-DD) for facts that expire, null for permanent facts"#;` (the closing raw string delimiter). Replace that entire block, keeping the `"#;` at the end:

```
## Output

Return ONLY valid JSON:

\`\`\`json
{
  "add": [
    {"fact_text": "...", "category": "identity", "permanent": true, "invalid_at": null, "entities": ["Person1", "Place1"], "relations": [{"s": "Person1", "r": "lives_in", "o": "Place1"}]}
  ],
  "update": [
    {"existing_fact_id": "...", "new_fact_text": "...", "category": "identity", "permanent": true, "invalid_at": null, "entities": [], "relations": []}
  ]
}
\`\`\`

If nothing worth remembering, return: {"add": [], "update": []}

Field reference:
- permanent: true for facts unlikely to change (name, background), false for time-bound facts
- invalid_at: ISO date string (YYYY-MM-DD) for facts that expire, null for permanent facts
- entities: list of key people, places, or things mentioned in the fact (optional, can be empty)
- relations: list of subject-relation-object triples connecting entities (optional, can be empty). Each has "s" (subject), "r" (relation verb), "o" (object)"#;
```

Note: The `\`\`\`json` and `\`\`\`` above represent actual triple backticks in the Rust raw string (shown escaped here to avoid markdown conflicts). The `"#;` at the very end closes the Rust raw string literal.

- [ ] **Step 2: Verify Rust compiles**

Run: `cd apps/api && cargo check 2>&1 | tail -3`
Expected: `Finished` with no errors (warnings OK)

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/services/prompts.rs
git commit -m "extend chat extraction prompt with optional entity/relation output"
```

---

### Task 7: Implement entity-hop retrieval

**Files:**
- Modify: `apps/api/evals/memory/src/locomo_adapter.py`

- [ ] **Step 1: Add entity-hop retrieval function**

Add after `_filter_relevant_facts`:

```python
def _entity_hop_retrieval(
    question: str,
    facts: list[dict],
    max_hops: int = 1,
    max_facts: int = 25,
) -> list[dict]:
    """Retrieve facts using entity-based graph traversal.

    1. Extract entities from question (keyword match against fact entities)
    2. Find seed facts mentioning those entities
    3. Expand by following entity links (up to max_hops)
    4. Fall back to keyword filter if no entity matches
    """
    q_lower = question.lower()

    # Build entity index: entity_name_lower -> list of fact indices
    entity_index: dict[str, list[int]] = {}
    for i, fact in enumerate(facts):
        for entity in fact.get("entities", []):
            entity_index.setdefault(entity.lower(), []).append(i)

    if not entity_index:
        # No entities extracted -- fall back to keyword filtering
        return _filter_relevant_facts(question, facts, max_facts=max_facts)

    # Step 1: Find entities mentioned in the question
    seed_entities: set[str] = set()
    for entity_name in entity_index:
        if entity_name in q_lower:
            seed_entities.add(entity_name)

    if not seed_entities:
        # No entity match -- fall back to keyword filtering
        return _filter_relevant_facts(question, facts, max_facts=max_facts)

    # Step 2: Collect seed facts
    seen_indices: set[int] = set()
    for entity in seed_entities:
        for idx in entity_index.get(entity, []):
            seen_indices.add(idx)

    # Step 3: Hop expansion
    current_entities = seed_entities.copy()
    for _hop in range(max_hops):
        # Collect new entities from current facts
        new_entities: set[str] = set()
        for idx in seen_indices:
            for entity in facts[idx].get("entities", []):
                ent_lower = entity.lower()
                if ent_lower not in current_entities:
                    new_entities.add(ent_lower)

        if not new_entities:
            break

        # Find facts mentioning new entities
        new_indices: set[int] = set()
        for entity in new_entities:
            for idx in entity_index.get(entity, []):
                if idx not in seen_indices:
                    new_indices.add(idx)

        seen_indices.update(new_indices)
        current_entities.update(new_entities)

    result = [facts[i] for i in sorted(seen_indices)]

    # If we got too many, apply relevance ranking
    if len(result) > max_facts:
        result = _filter_relevant_facts(question, result, max_facts=max_facts)

    # If entity hop found very few, supplement with keyword matches
    if len(result) < 5:
        keyword_results = _filter_relevant_facts(question, facts, max_facts=max_facts - len(result))
        seen_ids = {f.get("id") for f in result}
        for f in keyword_results:
            if f.get("id") not in seen_ids:
                result.append(f)
                if len(result) >= max_facts:
                    break

    return result
```

- [ ] **Step 2: Wire entity-hop into QA loop**

In the QA loop (Task 5 Step 2), replace `_filter_relevant_facts` with `_entity_hop_retrieval`:

```python
                # Filter facts relevant to this question (entity-hop + keyword fallback)
                relevant_facts = _entity_hop_retrieval(question, accumulated_facts)
                context = _group_facts_for_context(relevant_facts, current_date=last_date)
```

- [ ] **Step 3: Verify import**

Run: `cd apps/api/evals/memory && uv run python -c "from src.locomo_adapter import _entity_hop_retrieval; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add apps/api/evals/memory/src/locomo_adapter.py
git commit -m "add entity-hop retrieval for multi-hop QA in LoCoMo"
```

---

## Chunk 4: Run Benchmarks and Validate

### Task 9: Run LoCoMo smoke test (1 sample)

**Files:**
- No file changes -- run only

**Prerequisites:** Dev server running (`cd apps/api && wrangler dev`), LoCoMo data downloaded.

- [ ] **Step 1: Clear any stale LoCoMo caches**

```bash
rm -f apps/api/evals/memory/data/locomo_extraction_cache.jsonl
rm -f apps/api/evals/memory/data/locomo_qa_cache.jsonl
```

- [ ] **Step 2: Run 1-sample smoke test**

```bash
cd apps/api/evals/memory
uv run python -m src.locomo_adapter --live --locomo-samples 1
```

Expected: Should complete without errors. Watch for:
- Facts extracted > 0 (confirms adapter fix works)
- Overall F1 > 0 (confirms QA works)
- Rate limit retries happening and succeeding (confirms retry wrapper)

- [ ] **Step 3: Check extraction count and per-category breakdown**

If `Facts extracted: 0`, the adapter is still broken -- debug the `add`/`update` processing.
If F1 is very low (<0.1), check if the QA prompt is getting the right context.

---

### Task 10: Run full LoCoMo benchmark (all samples)

**Files:**
- No file changes -- run only

- [ ] **Step 1: Run full benchmark**

```bash
cd apps/api/evals/memory
uv run python -m src.locomo_adapter --live --locomo-samples 10
```

This will take a while due to rate limits (expect 20-40 minutes). The retry wrapper handles throttling automatically.

- [ ] **Step 2: Record results**

Note the overall F1 and per-category F1 scores. Target:
- Overall F1 > 65 (beats Mem0's 58.44)
- Multi-hop F1 > 55 (entity-hop should help here)

- [ ] **Step 3: Run the full report**

```bash
cd apps/api/evals/memory
uv run python -m src.run_all --layer report
```

This aggregates all layers (retrieval, synthesis, temporal, downstream, chat extraction, LoCoMo) into the comparison table.

---

### Task 11: Run chat extraction regression check

**Files:**
- No file changes -- run only

- [ ] **Step 1: Clear chat extraction cache (prompt changed in prompts.rs)**

```bash
rm -f apps/api/evals/memory/data/chat_extraction_cache.jsonl
```

- [ ] **Step 2: Re-run chat extraction against live API**

```bash
cd apps/api/evals/memory
uv run python -m src.eval_chat_extraction --live
```

May need two runs due to Groq rate limits (second run picks up from cache).

- [ ] **Step 3: Verify no regression**

Expected: recall >= 1.0, precision >= 0.85, operation accuracy >= 0.98.
If precision dropped, the entity/relation fields in the prompt may be causing the LLM to over-extract -- may need to make entities truly optional in the prompt wording.

- [ ] **Step 4: Final commit with results**

```bash
git add apps/api/evals/memory/src/locomo_adapter.py apps/api/src/services/prompts.rs
git commit -m "LoCoMo benchmark results and chat extraction regression check"
```

---

## File Summary

| File | Action | Task |
|------|--------|------|
| `apps/api/evals/memory/src/locomo_adapter.py` | Major rewrite | Tasks 1-2, 4-5, 7-8 |
| `apps/api/src/services/prompts.rs` | Extend output format | Task 6 |
| `apps/api/evals/memory/.gitignore` | Add locomo data | Task 3 |
| `apps/api/evals/memory/data/locomo10.json` | Download | Task 3 |

## Run Notes

- **Dev server must be running** for Tasks 9-11: `cd apps/api && wrangler dev`
- **Spec notes `memory.rs` changes** but entities are kept in the Python adapter only (per spec Section 3d: "No D1 schema changes"). The Rust API already ignores unknown JSON fields, so no `memory.rs` changes needed.
- **`sentence-transformers` is already in pyproject.toml** (listed in dependencies). No additional install needed.
- **Groq rate limits** (12k TPM free tier): the retry wrapper handles this, but full LoCoMo runs will be slow. Consider running overnight or upgrading Groq tier.
- **Caching**: All API responses are cached in JSONL files. Re-runs skip cached entries. Clear caches when prompts change.
- **Entity extraction quality**: The LLM may not consistently produce entities on the first try. If entity-hop retrieval falls back to keyword filtering for most questions, the entity prompt may need examples. Check `n_facts_filtered` in QA cache entries.
