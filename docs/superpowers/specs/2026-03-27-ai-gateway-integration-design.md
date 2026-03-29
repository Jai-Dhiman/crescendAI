# AI Gateway Integration Design

**Date:** 2026-03-27
**Status:** Approved
**Scope:** Route all LLM calls through Cloudflare AI Gateway, migrate Workers AI to optimal models, add shadow benchmarking for Groq vs Workers AI comparison.

## Context

CrescendAI's API Worker makes LLM calls to three providers (Anthropic, Groq, Workers AI) via direct HTTP. There is no fallback logic, no cost observability, no rate limiting, and no caching. OpenRouter was planned as a fallback but never implemented.

This design adds Cloudflare AI Gateway as a proxy layer between the Worker and all LLM providers, bringing observability, caching, rate limiting, and automatic Groq-to-Workers-AI fallback -- with minimal code change.

## Decisions

1. **No Anthropic fallback.** Teacher voice + tool_use is the product. If Anthropic is down, fail gracefully rather than serve degraded quality from a weaker model.
2. **Groq falls back to Workers AI (same model).** Groq runs Llama 3.3 70B; Workers AI runs the same model (FP8 quantized). Transparent fallback, identical output quality.
3. **Workers AI model migration.** Background tasks (titles, goals, memory extraction) move from Llama 3.3 70B to Qwen3-30b-a3b-fp8 (MoE, 3B active params, 7x cheaper output tokens).
4. **OpenRouter dropped.** Never implemented, no longer planned.
5. **Two gateways** for separated cost tracking: `crescendai-teacher` (Anthropic) and `crescendai-background` (Groq + Workers AI).
6. **Universal 60s cache TTL.** Cache miss cost is zero (passthrough). Any hit saves tokens.
7. **Conservative rate limiting.** 100 req/min per gateway. Catches bugs, not users.
8. **Shadow benchmarking.** 10% of subagent calls fire a parallel request to Workers AI to compare latency head-to-head with Groq. If Workers AI is within 200ms on p95, drop Groq entirely.

## Gateway Topology

### Two Gateways

| Gateway | Name | Providers | Rate Limit | Caching |
|---------|------|-----------|------------|---------|
| `crescendai-teacher` | Teacher path | Anthropic only (no fallback) | 100 req/min (sliding window) | 60s TTL |
| `crescendai-background` | Subagent + cheap tasks | Groq -> Workers AI Llama 3.3 70B (fallback), Workers AI Qwen3-30b | 100 req/min (sliding window) | 60s TTL |

### Provider Roles

| Provider | Role | Model | Gateway | Fallback |
|----------|------|-------|---------|----------|
| **Anthropic** | Teacher (chat, tool_use, streaming, elaboration, synthesis) | Claude Sonnet 4.6 | `crescendai-teacher` | None |
| **Groq** | Subagent analysis, memory synthesis | Llama 3.3 70B Versatile | `crescendai-background` | Workers AI Llama 3.3 70B FP8 |
| **Workers AI** | Titles, goal extraction, memory extraction | Qwen3-30b-a3b-fp8 | `crescendai-background` | None |

### Call Site Mapping

| Call Site | Function | Before | After |
|-----------|----------|--------|-------|
| `ask.rs:137` (subagent) | `call_groq` | Groq direct | Gateway -> Groq (fallback: Workers AI Llama 3.3) |
| `ask.rs:187` (teacher tool_use) | `call_anthropic_with_tools` | Anthropic direct | Gateway -> Anthropic |
| `ask.rs:436` (elaboration) | `call_anthropic` | Anthropic direct | Gateway -> Anthropic |
| `chat.rs:436` (streaming chat) | `call_anthropic_stream` | Anthropic direct | Gateway -> Anthropic |
| `chat.rs:675` (titles) | `call_workers_ai` | Llama 3.3 70B | Gateway -> Qwen3-30b-a3b-fp8 |
| `goals.rs:178` (extraction) | `call_workers_ai` | Llama 3.3 70B | Gateway -> Qwen3-30b-a3b-fp8 |
| `memory.rs:422,1033,1518` (extraction) | `call_workers_ai` | Llama 3.3 70B | Gateway -> Qwen3-30b-a3b-fp8 |
| `memory.rs:830` (synthesis) | `call_groq` | Groq direct | Gateway -> Groq (fallback: Workers AI Llama 3.3) |

## Gateway Client Abstraction

A thin `AiGateway` struct in `llm.rs` encapsulates gateway routing.

```rust
pub struct AiGateway {
    account_id: String,
    gateway_id: String,  // "crescendai-teacher" or "crescendai-background"
    cache_ttl: u32,      // seconds, 0 = no caching
}
```

### Responsibilities

- Construct gateway URL: `https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/{provider}/...`
- Attach caching headers (`cf-aig-cache-ttl`) to every request
- Read response metadata (`cf-aig-cache-status`, `cf-aig-step`) and log via `console_log!`
- For Groq calls, construct the universal endpoint fallback array (Groq primary -> Workers AI Llama 3.3 70B)

### Non-responsibilities

- No request/response format changes -- Anthropic calls keep native format, Groq keeps OpenAI-compatible format
- No retry logic -- the gateway handles retries
- No error transformation -- provider errors pass through unchanged

### Function Changes

| Function | Change |
|----------|--------|
| `call_groq` | URL swaps to gateway universal endpoint with fallback array |
| `call_anthropic` | URL swaps to `gateway.ai.cloudflare.com/.../anthropic/...` |
| `call_anthropic_with_tools` | Same URL swap |
| `call_anthropic_stream` | Same URL swap -- gateway proxies SSE streams transparently |
| `call_workers_ai` | Routes through gateway instead of `env.ai()` binding; gains `model: &str` parameter |

## Workers AI Model Migration

### Model Changes

| Call Site | Current Model | New Model | Cost Reduction (output $/M) |
|-----------|--------------|-----------|----------------------------|
| Titles | `llama-3.3-70b-instruct-fp8-fast` ($2.25) | `qwen3-30b-a3b-fp8` ($0.34) | 7x cheaper |
| Goal extraction | `llama-3.3-70b-instruct-fp8-fast` ($2.25) | `qwen3-30b-a3b-fp8` ($0.34) | 7x cheaper |
| Memory extraction (x3) | `llama-3.3-70b-instruct-fp8-fast` ($2.25) | `qwen3-30b-a3b-fp8` ($0.34) | 7x cheaper |

### Signature Change

`call_workers_ai` gains a `model: &str` parameter. Call sites pass a constant. The `env.ai()` binding is replaced with HTTP through the gateway so all Workers AI calls get unified observability.

### Prompt Validation

Qwen3-30b is a different model family. Existing prompts are simple structured extraction -- they should transfer. Manual spot-check of output quality for each call site during beta.

## Shadow Benchmarking

Compare Groq vs Workers AI latency to determine if Groq can be dropped entirely.

### Mechanism

For 10% of subagent calls (configurable), fire a parallel shadow request to Workers AI Llama 3.3 70B FP8. The shadow response content is discarded; only latency is logged. The shadow request runs concurrently with the primary Groq request (via `futures::join!` or similar) so it does not add latency to the caller, but it must be awaited (not fire-and-forget) to capture the timing.

```
Subagent call flow:
  1. Build prompt
  2. If SHADOW_BENCHMARK_ENABLED && rand() < SHADOW_BENCHMARK_PCT:
       futures::join!(groq_request, shadow_workers_ai_request)
       log shadow latency to console_log! (picked up by CF Observability)
       return groq result
  3. Else:
       send prompt to Groq via crescendai-background gateway
       return groq result
```

### Decision Criteria

After 2 weeks of beta traffic:
- Workers AI p95 within 200ms of Groq p95: drop Groq, remove `GROQ_API_KEY`, simplify to Anthropic + Workers AI
- Workers AI significantly slower: keep Groq, remove shadow code

## Configuration

### New `wrangler.toml` Vars

```toml
[vars]
CF_ACCOUNT_ID = "<account-id>"
SHADOW_BENCHMARK_ENABLED = "true"
SHADOW_BENCHMARK_PCT = "10"
```

No new secrets. Existing `GROQ_API_KEY` and `ANTHROPIC_API_KEY` pass through the gateway in request headers.

### Gateway Setup (One-Time, CF Dashboard or API)

1. Create `crescendai-teacher` gateway -- rate limit 100 req/min (sliding window), caching enabled, logging enabled
2. Create `crescendai-background` gateway -- rate limit 100 req/min (sliding window), caching enabled, logging enabled

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Anthropic returns 500 | Gateway retries once, then propagates error. Graceful failure message to student. |
| Anthropic is completely down | Same -- no fallback. Student sees "teacher unavailable." |
| Groq returns 500 | Gateway retries, then falls through to Workers AI Llama 3.3 70B. Transparent to caller. |
| Groq is completely down | Gateway falls through to Workers AI. Transparent to caller. |
| Gateway itself is down | Retry once against the provider's native URL (direct fallback in Rust code). |
| Rate limit hit (429 from gateway) | Log warning, return error to caller. Investigate -- something is wrong. |
| Workers AI model unavailable | Error propagated (cheapest tier, no fallback). |

## Observability

Every LLM call logs via `console_log!`:
- Provider used (from `cf-aig-step` header: 0 = primary, 1 = fallback)
- Cache hit/miss (from `cf-aig-cache-status` header)
- Gateway name (`teacher` vs `background`)
- Latency (measured in Rust)
- Shadow benchmark latency (when applicable)

All logs flow through existing CF Observability -> Sentry OTLP pipeline.

## What This Does NOT Change

- Anthropic request/response format (native tool_use, streaming SSE, cache_control blocks)
- Groq request/response format (OpenAI-compatible)
- Any business logic in ask.rs, chat.rs, memory.rs, goals.rs
- The `env.ai("AI")` binding remains available but is no longer the primary call path for Workers AI
- Authentication flow, D1 schema, R2 storage, Durable Objects
