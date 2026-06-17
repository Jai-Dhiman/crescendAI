# Implementation Notes — V6 Teacher on Workers AI

Issue #61. Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: tool-format.ts (Anthropic↔OpenAI translation)
Pure translation layer. `toOpenAIChatRequest` maps tool defs/tool_choice/content blocks; `toAnthropicResponse` maps back, JSON-parsing function.arguments into tool_use.input. Mixed [text, tool_use] assistant messages intentionally drop the leading text (content:null) — documented in the test; tool extraction unaffected. Review fix: wrapped JSON.parse in try/catch throwing explicit InferenceError on malformed tool-call args (project "explicit exception handling" standard).

## Task 2: gateway-client.ts (callModel)
`callModel(env, client, body)` branches on provider. anthropic: POST /anthropic/v1/messages, cf-aig-authorization. workers-ai: translate via tool-format, POST /workers-ai/v1/chat/completions, cf-aig-authorization + Authorization(CLOUDFLARE_API_TOKEN). Both throw InferenceError on non-2xx.

## Task 3: route-model.ts
Replaced GatewayClient/gatewayUrlVar with ModelClient {provider, model}. `routeModel(kind, env)` defaults to workers-ai (@cf/qwen/qwen3-30b-a3b-fp8); TEACHER_PROVIDER=anthropic returns Sonnet. ModelClient is defined identically in both route-model.ts and gateway-client.ts (documented OBS in plan; structurally compatible).

## Task 4: types.ts (additive)
Added AI_GATEWAY_ENDPOINT, AI_GATEWAY_TOKEN, TEACHER_PROVIDER? to Bindings. Old vars kept until Task 8.

## Task 5: phase1.ts
Removed local callAnthropicMessage; delegates to callModel + routeModel(kind, ctx.env). Review fix: widened the messages-array content union to include {type:"text",text} so response.content (which may carry text blocks from callModel) type-checks.

## Task 6: phase2.ts
Removed local Anthropic call; delegates to callModel. 3-attempt Zod repair loop, forced tool_choice, validation_error exhaustion, zodToJsonSchema($refStrategy:"none") all preserved byte-for-byte.

## Task 7: llm.ts
callWorkersAI/callAnthropic/callAnthropicStream all switched to AI_GATEWAY_ENDPOINT + cf-aig-authorization(AI_GATEWAY_TOKEN); x-api-key/ANTHROPIC_API_KEY removed. callAnthropicStream (SSE chat path) has no unit test — needs orchestrator live smoke (a 400 credit-balance-too-low is acceptable; a 401/AiGatewayError is a regression).

## Task 8: dead-var cleanup (scope expanded vs plan)
Removed AI_GATEWAY_TEACHER/AI_GATEWAY_BACKGROUND/ANTHROPIC_API_KEY from Bindings + updated all mocks. PLAN GAP FOUND: the plan listed 5 test files, but grep found 2 more references — `src/services/teacher.test.ts` (mock) and `src/services/chat.ts` (PRODUCTION). chat.ts line 128 gated conversation-title generation on `env.AI_GATEWAY_BACKGROUND` — a now-removed field; fixed to `env.AI_GATEWAY_ENDPOINT` (the var callWorkersAI actually uses). Without this fix title-gen would silently never fire AND typecheck would break.

## Pre-existing typecheck state
Branch carries 20 pre-existing typecheck errors unrelated to this work (wasm pkg decls, segment-loops import, node:fs/promises/node:path missing types, exercise-routing primitive_id, phase2.test CompoundBinding-vs-Phase2Binding, exercises.test overloads, harness/artifacts/index unused exports). This build added zero new errors. The phase2.test.ts CompoundBinding/Phase2Binding mismatch predates the branch (present at base b56205aa).
