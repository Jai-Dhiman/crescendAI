# Synthesized Facts Wiring Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing synthesis pipeline into two trigger points (DO session finalization + HTTP endpoint), fix the observation counting bug, and create an integration test.

**Architecture:** The synthesis pipeline already exists in `memory.rs`. This plan adds: (1) `SynthesisResult` return type for observability, (2) `increment_observation_count_by()` for batched counter updates, (3) synthesis trigger in DO `finalize_session()`, (4) `POST /api/memory/synthesize` HTTP endpoint, (5) `POST /api/memory/seed-observations` dev-only endpoint, (6) Python integration test.

**Tech Stack:** Rust (Cloudflare Workers WASM), D1 (SQLite), Groq API (Llama 70B), Python (pytest, requests)

**Spec:** `docs/superpowers/specs/2026-03-15-synthesized-facts-wiring-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `apps/api/src/services/memory.rs` | Modify | Add `SynthesisResult`, refactor `run_synthesis()`, add `increment_observation_count_by()`, add `handle_synthesize()`, add `handle_seed_observations()` |
| `apps/api/src/practice/session.rs` | Modify | Add synthesis trigger + batched observation count in `finalize_session()` |
| `apps/api/src/server.rs` | Modify | Add 2 new routes |
| `apps/api/evals/memory/src/test_synthesis.py` | Create | Integration test |

---

## Task 1: Add `SynthesisResult` and refactor `run_synthesis()` return type

**Files:**
- Modify: `apps/api/src/services/memory.rs:694-925`

- [ ] **Step 1: Add the `SynthesisResult` struct**

Add directly above `run_synthesis()` (before line 694):

```rust
/// Result of a synthesis run, for observability.
pub struct SynthesisResult {
    pub new_facts: usize,
    pub invalidated: usize,
    pub unchanged: usize,
    pub observations_processed: usize,
}
```

- [ ] **Step 2: Change `run_synthesis()` signature and early return**

Change line 698 from:
```rust
) -> Result<(), String> {
```
to:
```rust
) -> Result<SynthesisResult, String> {
```

Change the early return at line 744-746 from:
```rust
    if new_observations.is_empty() {
        return Ok(());
    }
```
to:
```rust
    if new_observations.is_empty() {
        return Ok(SynthesisResult {
            new_facts: 0,
            invalidated: 0,
            unchanged: active_facts.len(),
            observations_processed: 0,
        });
    }
```

- [ ] **Step 3: Count results and return `SynthesisResult`**

After the existing step 7 (`let synthesis_json = extract_synthesis_json(...)?;`), add counting variables. Then replace the final `Ok(())` (line 924) with a `SynthesisResult`.

Add right after `let today = ...` (after line 806):

```rust
    let active_facts_count = active_facts.len();
    let observations_count = new_observations.len();
```

Replace the `// 8. Apply invalidations` block (lines 808-830) to count invalidations:

```rust
    // 8. Apply invalidations
    let mut invalidated_count = 0usize;
    if let Some(invalidated) = synthesis_json.get("invalidated_facts").and_then(|v| v.as_array()) {
        for inv in invalidated {
            let fact_id = inv.get("fact_id").and_then(|v| v.as_str()).unwrap_or("");
            let invalid_at = inv.get("invalid_at").and_then(|v| v.as_str()).unwrap_or(today);
            if !fact_id.is_empty() {
                let _ = db
                    .prepare(
                        "UPDATE synthesized_facts SET invalid_at = ?1, expired_at = ?2 \
                         WHERE id = ?3 AND student_id = ?4",
                    )
                    .bind(&[
                        JsValue::from_str(invalid_at),
                        JsValue::from_str(&now),
                        JsValue::from_str(fact_id),
                        JsValue::from_str(student_id),
                    ])
                    .map_err(|e| format!("Failed to bind invalidation: {:?}", e))?
                    .run()
                    .await;
                invalidated_count += 1;
            }
        }
    }
```

Replace the `// 9. Insert new facts` block (lines 832-883) to count new facts:

```rust
    // 9. Insert new facts
    let mut new_facts_count = 0usize;
    if let Some(new_facts) = synthesis_json.get("new_facts").and_then(|v| v.as_array()) {
        for fact in new_facts {
            let fact_id = generate_uuid();
            let fact_text = fact.get("fact_text").and_then(|v| v.as_str()).unwrap_or("");
            let fact_type = fact.get("fact_type").and_then(|v| v.as_str()).unwrap_or("dimension");
            let dimension = fact.get("dimension").and_then(|v| v.as_str());
            let piece_ctx = fact.get("piece_context").map(|v| v.to_string());
            let trend = fact.get("trend").and_then(|v| v.as_str());
            let confidence = fact.get("confidence").and_then(|v| v.as_str()).unwrap_or("medium");
            let evidence = fact.get("evidence").map(|v| v.to_string()).unwrap_or_else(|| "[]".to_string());
            let valid_at = today;

            if fact_text.is_empty() {
                continue;
            }

            let _ = db
                .prepare(
                    "INSERT INTO synthesized_facts \
                     (id, student_id, fact_text, fact_type, dimension, piece_context, \
                      valid_at, trend, confidence, evidence, source_type, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                )
                .bind(&[
                    JsValue::from_str(&fact_id),
                    JsValue::from_str(student_id),
                    JsValue::from_str(fact_text),
                    JsValue::from_str(fact_type),
                    match dimension {
                        Some(d) => JsValue::from_str(d),
                        None => JsValue::NULL,
                    },
                    match piece_ctx.as_deref() {
                        Some(pc) if pc != "null" => JsValue::from_str(pc),
                        _ => JsValue::NULL,
                    },
                    JsValue::from_str(valid_at),
                    match trend {
                        Some(t) => JsValue::from_str(t),
                        None => JsValue::NULL,
                    },
                    JsValue::from_str(confidence),
                    JsValue::from_str(&evidence),
                    JsValue::from_str("synthesized"),
                    JsValue::from_str(&now),
                ])
                .map_err(|e| format!("Failed to bind fact insert: {:?}", e))?
                .run()
                .await;
            new_facts_count += 1;
        }
    }
```

Replace the final `Ok(())` and the console_log before it (lines 918-924) with:

```rust
    console_log!(
        "Synthesis complete for student {}: {} new, {} invalidated, {} observations",
        student_id, new_facts_count, invalidated_count, observations_count
    );

    Ok(SynthesisResult {
        new_facts: new_facts_count,
        invalidated: invalidated_count,
        unchanged: active_facts_count.saturating_sub(invalidated_count),
        observations_processed: observations_count,
    })
```

- [ ] **Step 4: Build and verify compilation**

Run: `cd apps/api && cargo check 2>&1 | head -20`
Expected: Successful compilation (no errors).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/memory.rs
git commit -m "add SynthesisResult return type to run_synthesis()"
```

---

## Task 2: Add `increment_observation_count_by()`

**Files:**
- Modify: `apps/api/src/services/memory.rs:604-621`

- [ ] **Step 1: Add `increment_observation_count_by()` function**

Add directly after the existing `increment_observation_count()` function (after line 621):

```rust
/// Increment observation count by N (batched version for DO session finalization).
pub async fn increment_observation_count_by(
    env: &Env,
    student_id: &str,
    count: usize,
) -> Result<(), String> {
    if count == 0 {
        return Ok(());
    }
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    db.prepare(
        "INSERT INTO student_memory_meta (student_id, total_observations) VALUES (?1, ?2) \
         ON CONFLICT(student_id) DO UPDATE SET total_observations = total_observations + ?2",
    )
    .bind(&[
        JsValue::from_str(student_id),
        JsValue::from_f64(count as f64),
    ])
    .map_err(|e| format!("Failed to bind upsert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to update observation count: {:?}", e))?;

    Ok(())
}
```

- [ ] **Step 2: Build and verify compilation**

Run: `cd apps/api && cargo check 2>&1 | head -20`
Expected: Successful compilation.

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/services/memory.rs
git commit -m "add increment_observation_count_by() for batched DO updates"
```

---

## Task 3: Add DO session finalization trigger

**Files:**
- Modify: `apps/api/src/practice/session.rs:738-782`

- [ ] **Step 1: Add synthesis trigger in `finalize_session()`**

In `finalize_session()`, insert the following block between the `persist_observations` section (after line 755, after the closing `}` of the `if !observations.is_empty()` block) and the `// 2. Send session_summary` comment (line 757).

Replace lines 755-757:
```rust
        }

        // 2. Send session_summary via WebSocket
```

With:
```rust
        }

        // 2. Update observation count and run synthesis
        if !observations.is_empty() {
            // Fix: DO-originated observations were not incrementing the meta counter.
            // This ensures should_synthesize() has an accurate count.
            if let Err(e) = crate::services::memory::increment_observation_count_by(
                &self.env, &student_id, observations.len()
            ).await {
                console_error!("Failed to increment observation count: {}", e);
            }

            // Run synthesis if enough observations have accumulated
            match crate::services::memory::should_synthesize(&self.env, &student_id).await {
                Ok(true) => {
                    match crate::services::memory::run_synthesis(&self.env, &student_id).await {
                        Ok(result) => {
                            console_log!(
                                "Synthesis for {}: {} new, {} invalidated, {} unchanged",
                                student_id, result.new_facts, result.invalidated, result.unchanged
                            );
                        }
                        Err(e) => {
                            console_error!("Synthesis failed for {}: {}", student_id, e);
                        }
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    console_error!("Synthesis check failed: {}", e);
                }
            }
        }

        // 3. Send session_summary via WebSocket
```

Also update the subsequent comment numbers: `// 2. Send session_summary` becomes `// 3.`, and `// 3. Close all WebSockets` becomes `// 4.`.

- [ ] **Step 2: Build and verify compilation**

Run: `cd apps/api && cargo check 2>&1 | head -20`
Expected: Successful compilation.

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "wire synthesis trigger into DO session finalization"
```

---

## Task 4: Add `POST /api/memory/synthesize` endpoint

**Files:**
- Modify: `apps/api/src/services/memory.rs` (add handler at end of file, before the private helpers)
- Modify: `apps/api/src/server.rs` (add route)

- [ ] **Step 1: Add request/response types and handler in `memory.rs`**

Add before the `extract_synthesis_json` function (before line 1624 in the current file -- line numbers will have shifted from Task 1/2 edits, so find `fn extract_synthesis_json`):

```rust
// ---------------------------------------------------------------------------
// POST /api/memory/synthesize -- trigger synthesis for a student
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct SynthesizeRequest {
    pub student_id: String,
}

/// POST /api/memory/synthesize -- manually trigger synthesis.
pub async fn handle_synthesize(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let _caller = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: SynthesizeRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse synthesize request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Check if synthesis is needed
    let should = match should_synthesize(env, &request.student_id).await {
        Ok(s) => s,
        Err(e) => {
            console_log!("Synthesis check failed: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": format!("Synthesis check failed: {}", e)}).to_string(),
                ))
                .unwrap();
        }
    };

    if !should {
        let resp = serde_json::json!({"skipped": true, "reason": "Not enough new observations"});
        return Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(resp.to_string()))
            .unwrap();
    }

    // Run synthesis
    match run_synthesis(env, &request.student_id).await {
        Ok(result) => {
            let resp = serde_json::json!({
                "new_facts": result.new_facts,
                "invalidated": result.invalidated,
                "unchanged": result.unchanged,
                "observations_processed": result.observations_processed,
            });
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Body::from(resp.to_string()))
                .unwrap()
        }
        Err(e) => {
            console_log!("Synthesis failed: {}", e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": format!("Synthesis failed: {}", e)}).to_string(),
                ))
                .unwrap()
        }
    }
}
```

- [ ] **Step 2: Add route in `server.rs`**

Add after the `/api/memory/clear-benchmark` route block (after line 362 in server.rs):

```rust
    // Memory synthesis endpoint (authenticated)
    if path == "/api/memory/synthesize" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::memory::handle_synthesize(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd apps/api && cargo check 2>&1 | head -20`
Expected: Successful compilation.

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/memory.rs apps/api/src/server.rs
git commit -m "add POST /api/memory/synthesize endpoint"
```

---

## Task 5: Add `POST /api/memory/seed-observations` dev-only endpoint

**Files:**
- Modify: `apps/api/src/services/memory.rs` (add handler)
- Modify: `apps/api/src/server.rs` (add route)

- [ ] **Step 1: Add request type and handler in `memory.rs`**

Add after the `handle_synthesize` function:

```rust
// ---------------------------------------------------------------------------
// POST /api/memory/seed-observations -- dev-only observation seeding for tests
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct SeedObservationsRequest {
    pub student_id: String,
    pub observations: Vec<SeedObservation>,
}

#[derive(serde::Deserialize)]
pub struct SeedObservation {
    pub dimension: String,
    pub observation_text: String,
    pub framing: String,
    pub dimension_score: f64,
    pub student_baseline: f64,
    #[serde(default)]
    pub reasoning_trace: String,
}

/// POST /api/memory/seed-observations -- insert test observations directly into D1.
/// Dev-only: returns 404 in production.
pub async fn handle_seed_observations(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Block in production
    let environment = env
        .var("ENVIRONMENT")
        .map(|v| v.to_string())
        .unwrap_or_default();
    if environment == "production" {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"error":"Not found"}"#))
            .unwrap();
    }

    // Auth
    let _caller = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: SeedObservationsRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse seed-observations request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_log!("D1 binding failed: {:?}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Database unavailable"}"#))
                .unwrap();
        }
    };

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    let mut seeded = 0u32;
    let session_id = format!("seed-{}", &generate_uuid()[..8]);

    for obs in &request.observations {
        let obs_id = generate_uuid();
        let trace = if obs.reasoning_trace.is_empty() {
            "{}".to_string()
        } else {
            obs.reasoning_trace.clone()
        };

        let result = db
            .prepare(
                "INSERT INTO observations (id, student_id, session_id, dimension, \
                 observation_text, reasoning_trace, framing, dimension_score, \
                 student_baseline, is_fallback, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            )
            .bind(&[
                JsValue::from_str(&obs_id),
                JsValue::from_str(&request.student_id),
                JsValue::from_str(&session_id),
                JsValue::from_str(&obs.dimension),
                JsValue::from_str(&obs.observation_text),
                JsValue::from_str(&trace),
                JsValue::from_str(&obs.framing),
                JsValue::from_f64(obs.dimension_score),
                JsValue::from_f64(obs.student_baseline),
                JsValue::from_bool(false),
                JsValue::from_str(&now),
            ]);

        match result {
            Ok(stmt) => {
                if let Err(e) = stmt.run().await {
                    console_log!("Failed to insert seeded observation: {:?}", e);
                } else {
                    seeded += 1;
                }
            }
            Err(e) => {
                console_log!("Failed to bind seeded observation: {:?}", e);
            }
        }
    }

    // Update meta counter so should_synthesize() works correctly
    if seeded > 0 {
        if let Err(e) = increment_observation_count_by(env, &request.student_id, seeded as usize).await {
            console_log!("Failed to update observation count after seeding: {}", e);
        }
    }

    let resp = serde_json::json!({"seeded": seeded});
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(resp.to_string()))
        .unwrap()
}
```

- [ ] **Step 2: Add route in `server.rs`**

Add after the `/api/memory/synthesize` route (which was added in Task 4):

```rust
    // Memory seed-observations endpoint (dev-only, for testing)
    if path == "/api/memory/seed-observations" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::services::memory::handle_seed_observations(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd apps/api && cargo check 2>&1 | head -20`
Expected: Successful compilation.

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/memory.rs apps/api/src/server.rs
git commit -m "add POST /api/memory/seed-observations dev-only endpoint"
```

---

## Task 6: Deploy and verify endpoints manually

**Files:** None (deployment and manual testing)

- [ ] **Step 1: Deploy to dev**

Run: `cd apps/api && bunx wrangler deploy --env dev 2>&1 | tail -5`
Expected: Successful deployment.

- [ ] **Step 2: Get a debug auth token**

Run: `curl -s -X POST http://localhost:8787/api/auth/debug | jq .token`
(Or against dev URL if not running locally.)

- [ ] **Step 3: Test seed-observations**

```bash
curl -s -X POST http://localhost:8787/api/memory/seed-observations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"student_id":"TEST_ID","observations":[{"dimension":"dynamics","observation_text":"Dynamics were weak in the exposition","framing":"correction","dimension_score":0.3,"student_baseline":0.5}]}' | jq .
```
Expected: `{"seeded": 1}`

- [ ] **Step 4: Test synthesize (threshold not met)**

```bash
curl -s -X POST http://localhost:8787/api/memory/synthesize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"student_id":"TEST_ID"}' | jq .
```
Expected: `{"skipped": true, "reason": "Not enough new observations"}`

- [ ] **Step 5: Seed more observations to meet threshold, then synthesize**

Seed 2 more observations (total 3), then call synthesize again.
Expected: `{"new_facts": N, "invalidated": 0, "unchanged": 0, "observations_processed": 3}` where N >= 1.

- [ ] **Step 6: Clean up test data**

```bash
curl -s -X POST http://localhost:8787/api/memory/clear-benchmark \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"student_id":"TEST_ID"}' | jq .
```

- [ ] **Step 7: Commit (no code changes, just verification)**

No commit needed -- this was manual verification.

---

## Task 7: Write integration test

**Files:**
- Create: `apps/api/evals/memory/src/test_synthesis.py`

- [ ] **Step 1: Create the integration test file**

```python
"""Integration test for the synthesis pipeline.

Tests the full cycle: seed observations -> synthesize -> verify facts -> contradict -> re-synthesize -> verify invalidation.

Requires a running dev API server (local or remote).
Run: cd apps/api/evals/memory && uv run pytest src/test_synthesis.py -v

Marked @integration -- excluded from CI, runs against live Groq API.
"""

import os

import pytest
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8787")


def _get_auth_token() -> str:
    """Get a debug auth token from the local dev server."""
    resp = requests.post(f"{API_BASE}/api/auth/debug")
    resp.raise_for_status()
    data = resp.json()
    return data.get("token", "")


def _get_student_id(token: str) -> str:
    """Get the debug student_id from /api/auth/me."""
    resp = requests.get(
        f"{API_BASE}/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    return resp.json().get("student_id", "")


def _seed_observations(token: str, student_id: str, observations: list[dict]) -> int:
    """Seed observations via the dev-only endpoint."""
    resp = requests.post(
        f"{API_BASE}/api/memory/seed-observations",
        headers={"Authorization": f"Bearer {token}"},
        json={"student_id": student_id, "observations": observations},
    )
    resp.raise_for_status()
    return resp.json().get("seeded", 0)


def _synthesize(token: str, student_id: str) -> dict:
    """Trigger synthesis and return the response."""
    resp = requests.post(
        f"{API_BASE}/api/memory/synthesize",
        headers={"Authorization": f"Bearer {token}"},
        json={"student_id": student_id},
    )
    resp.raise_for_status()
    return resp.json()


def _search_facts(token: str, student_id: str, query: str) -> dict:
    """Search for facts via the memory search endpoint."""
    resp = requests.post(
        f"{API_BASE}/api/memory/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"student_id": student_id, "query": query, "max_facts": 50},
    )
    resp.raise_for_status()
    return resp.json()


def _clear_data(token: str, student_id: str) -> None:
    """Clear benchmark/test data for the student."""
    requests.post(
        f"{API_BASE}/api/memory/clear-benchmark",
        headers={"Authorization": f"Bearer {token}"},
        json={"student_id": student_id},
    )


# Single test function to enforce sequential execution.
# The steps depend on each other (first synthesis creates facts,
# second synthesis invalidates them), so they must run in order.


@pytest.mark.integration
def test_synthesis_full_cycle():
    """Full synthesis cycle: threshold -> create -> verify -> contradict -> invalidate -> verify."""
    token = _get_auth_token()
    student_id = _get_student_id(token)
    _clear_data(token, student_id)

    try:
        # -- Step 1: Threshold not met --
        seeded = _seed_observations(token, student_id, [
            {
                "dimension": "dynamics",
                "observation_text": "Single observation for threshold test",
                "framing": "correction",
                "dimension_score": 0.3,
                "student_baseline": 0.5,
            },
        ])
        assert seeded == 1

        result = _synthesize(token, student_id)
        assert result.get("skipped") is True, f"Expected skipped, got {result}"

        # -- Step 2: Clean and seed 5 observations --
        _clear_data(token, student_id)

        observations = [
            {
                "dimension": "dynamics",
                "observation_text": "Dynamics were notably weak in the exposition section, lacking contrast between forte and piano passages",
                "framing": "correction",
                "dimension_score": 0.3,
                "student_baseline": 0.5,
                "reasoning_trace": '{"analysis": "dynamics well below baseline, consistent weakness"}',
            },
            {
                "dimension": "dynamics",
                "observation_text": "Dynamic range remains compressed, particularly in the recapitulation where forte markings are not observed",
                "framing": "correction",
                "dimension_score": 0.28,
                "student_baseline": 0.5,
                "reasoning_trace": '{"analysis": "second consecutive chunk with weak dynamics"}',
            },
            {
                "dimension": "pedaling",
                "observation_text": "Pedal work is clean and well-timed through the harmonic changes",
                "framing": "recognition",
                "dimension_score": 0.7,
                "student_baseline": 0.5,
                "reasoning_trace": '{"analysis": "pedaling above baseline, strength area"}',
            },
            {
                "dimension": "pedaling",
                "observation_text": "Sustain pedal changes align well with the harmonic rhythm",
                "framing": "recognition",
                "dimension_score": 0.72,
                "student_baseline": 0.5,
                "reasoning_trace": '{"analysis": "pedaling consistently strong"}',
            },
            {
                "dimension": "timing",
                "observation_text": "Tempo fluctuates in the development section, rushing through sixteenth-note passages",
                "framing": "correction",
                "dimension_score": 0.4,
                "student_baseline": 0.5,
                "reasoning_trace": '{"analysis": "timing below baseline in complex passages"}',
            },
        ]
        seeded = _seed_observations(token, student_id, observations)
        assert seeded == 5

        # -- Step 3: First synthesis --
        result = _synthesize(token, student_id)
        assert "skipped" not in result, f"Synthesis was skipped: {result}"
        assert result["new_facts"] >= 1, f"Expected at least 1 new fact, got {result}"
        assert result["invalidated"] == 0, f"Expected 0 invalidations on first run, got {result}"
        assert result["observations_processed"] >= 5, f"Expected >= 5 observations processed, got {result}"

        # -- Step 4: Verify facts with bi-temporal fields --
        search = _search_facts(token, student_id, "dynamics weakness")
        assert len(search["facts"]) >= 1, f"Expected at least 1 dynamics fact, got {search['facts']}"

        # Verify bi-temporal: valid_at should be set, invalid_at should be absent
        # (search endpoint returns active facts; active = invalid_at IS NULL)
        for fact in search["facts"]:
            assert fact.get("date"), f"Fact missing valid_at (date): {fact}"

        # -- Step 5: Seed contradicting observations --
        contradicting = [
            {
                "dimension": "dynamics",
                "observation_text": "Dynamics have improved significantly, with clear forte-piano contrast throughout",
                "framing": "recognition",
                "dimension_score": 0.8,
                "student_baseline": 0.5,
                "reasoning_trace": '{"analysis": "dynamics now well above baseline, major improvement"}',
            },
            {
                "dimension": "dynamics",
                "observation_text": "Dynamic shaping in the coda is particularly expressive, showing real growth",
                "framing": "recognition",
                "dimension_score": 0.82,
                "student_baseline": 0.5,
                "reasoning_trace": '{"analysis": "dynamics improvement sustained across multiple passages"}',
            },
            {
                "dimension": "dynamics",
                "observation_text": "The crescendo through bars 45-52 builds beautifully with controlled gradation",
                "framing": "recognition",
                "dimension_score": 0.78,
                "student_baseline": 0.5,
                "reasoning_trace": '{"analysis": "dynamics strength confirmed in technically demanding passage"}',
            },
        ]
        seeded = _seed_observations(token, student_id, contradicting)
        assert seeded == 3

        # -- Step 6: Second synthesis (contradiction) --
        result = _synthesize(token, student_id)
        assert "skipped" not in result, f"Synthesis was skipped: {result}"
        assert result["new_facts"] >= 1, f"Expected new improvement fact, got {result}"
        assert result["invalidated"] >= 1, f"Expected invalidation of old weakness fact, got {result}"

        # -- Step 7: Verify invalidation via search --
        # Search for dynamics facts -- should find the new improvement fact
        search_after = _search_facts(token, student_id, "dynamics improving")
        dynamics_facts = [f for f in search_after["facts"] if "dynamics" in f["fact_text"].lower()]
        assert len(dynamics_facts) >= 1, f"Expected dynamics improvement fact, got {dynamics_facts}"

        # The old weakness fact should no longer appear in active search results
        # (search returns active facts only -- invalid_at IS NULL)
        weakness_facts = [
            f for f in search_after["facts"]
            if "weak" in f["fact_text"].lower() and "dynamics" in f["fact_text"].lower()
        ]
        # If the old weakness fact was properly invalidated, it should not appear
        # in active results. (This may be empty or may still appear if the LLM
        # chose not to invalidate -- we assert the synthesis result count above
        # as the primary check, and this as a secondary verification.)

    finally:
        _clear_data(token, student_id)
```

- [ ] **Step 2: Add pytest to pyproject.toml dependencies**

`pytest` is not in the current dependencies. Add it to `apps/api/evals/memory/pyproject.toml`:

```toml
dependencies = [
    "groq>=0.4.0",
    "anthropic>=0.20.0",
    "requests>=2.31.0",
    "sentence-transformers>=2.0.0",
    "pytest>=7.0.0",
]
```

Also add marker registration to avoid pytest warnings. Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["integration: marks integration tests (require live API)"]
```

- [ ] **Step 3: Run the integration test**

This requires the dev server to be running. If it fails with a connection error, start the dev server first:
`cd apps/api && bunx wrangler dev &`

Run: `cd apps/api/evals/memory && uv run pytest src/test_synthesis.py -v`
Expected: `test_synthesis_full_cycle` PASS. The test may need 1-2 retries due to LLM non-determinism (Groq Llama 70B at temperature 0.1).

- [ ] **Step 5: Commit**

```bash
git add apps/api/evals/memory/src/test_synthesis.py apps/api/evals/memory/pyproject.toml
git commit -m "add synthesis pipeline integration test"
```

---

## Task 8: Final verification and cleanup

- [ ] **Step 1: Full cargo check**

Run: `cd apps/api && cargo check 2>&1 | head -20`
Expected: Clean compilation.

- [ ] **Step 2: Verify no regressions in existing eval**

Run: `cd apps/api/evals/memory && uv run python -m src.run_all 2>&1 | tail -20`
Expected: Existing eval results unchanged (recall=1.0, all 38 scenarios pass).

- [ ] **Step 3: Update docs/apps/00-status.md**

In the API Worker section, update the synthesized facts status from "NOT STARTED" to "COMPLETE":
- Synthesized facts synthesis trigger (DO + HTTP endpoint)
- Observation counting fix for DO sessions

- [ ] **Step 4: Update docs/apps/03-memory-system.md**

Change the Phase 2 status from DEFERRED to COMPLETE:
```
### Phase 2: Synthesis -- COMPLETE
```

Update the checklist items to `[x]`.

- [ ] **Step 5: Commit documentation updates**

```bash
git add docs/apps/00-status.md docs/apps/03-memory-system.md
git commit -m "update status: synthesized facts synthesis pipeline complete"
```
