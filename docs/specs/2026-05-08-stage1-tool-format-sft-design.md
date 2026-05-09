# Stage 1 Tool-Format SFT Training-Example Spec

**Goal:** Produce the locked specification, authoring tooling, and evaluation harness for ~2K Stage 1 training examples that teach Qwen3.6-35B-A3B to natively emit chat-template-native Anthropic-shaped `tool_use` blocks for the production 6-tool palette, with disciplined when-to-call behavior.

**Not in scope:**
- Generating the actual ~2K corpus (running `cli.py distill` repeatedly, hand-authoring ~600 negatives + ~50 matched-contrast pairs) -- execution work, follows this plan.
- The LoRA SFT training run itself (Unsloth invocation, hyperparameters, MoE telemetry to Trackio) -- separate plan after Stage 0 dossier completes and tooling lands.
- Stage 0 base-model probes (covered in `docs/specs/2026-05-07-stage0-capability-probe-design.md`, amended 2026-05-08).
- Modifying `apps/api/src/services/tool-processor.ts` or `teacher.ts` -- production code is the source of truth Stage 1 trains *toward*; no production changes here.
- Stages 2-4 data work (briefing-shaped SFT, capability SFT, DPO).

## Problem

`apps/evals/teacher_model/TRAINING_PLAN.md` Section 4 specifies Stage 1 as "~2K examples, ~30% negative, LoRA SFT" but does not lock the example shape, the authoring methodology, the per-tool argument-coverage matrix, or the eval acceptance criteria. Without those locks, data authors cannot start producing examples and the carry-forward LoRA constraint (one adapter through Stages 1-4) means any format drift introduced now propagates through every subsequent stage.

Three concrete blockers:

1. **No locked output format.** The training plan flags Open Question #2 ("`qwen3_coder` parser format vs Qwen3.5"). Until the chat template is verified and the assistant-turn shape is fixed, any example authored is at risk of training a serving/training mismatch -- the silent failure mode where eval passes but production parsing fails.
2. **No authoring pipeline.** The 6-tool palette in `apps/api/src/services/tool-processor.ts` defines Zod schemas and Anthropic tool descriptions, but there is no Python pipeline that (a) generates candidate examples via Sonnet distillation, (b) validates them against those schemas, (c) tracks coverage of every enum value and conditional-required combination.
3. **No evaluation harness.** Stage 1's "done" criterion is undefined. Without a held-out set + per-metric thresholds + serving-runtime parse rate measurement, "Stage 1 ready for Stage 2" is a judgment call that contaminates the carry-forward adapter.

The existing `apps/evals/teacher_model/tool_format.py` was a TS-bridging stopgap covering only `create_exercise` in OpenAI/ChatML format -- not a Stage 1 authoring solution and now obsoleted by the chat-template-native approach.

## Solution (from the user's perspective)

A data engineer runs:

```bash
# One-time, after Stage 0 pin-tokenizer completes:
uv run python -m teacher_model.stage1 holdout --frac 0.12 --seed 42

# Iteratively until coverage is satisfied:
uv run python -m teacher_model.stage1 distill --shape synthesis --n 200
uv run python -m teacher_model.stage1 distill --shape chat --n 100
uv run python -m teacher_model.stage1 coverage  # reports unfilled cells
# ... repeat distill until coverage.is_satisfied()

# Hand-author negatives + pairs into apps/evals/teacher_model/stage1/negatives/
uv run python -m teacher_model.stage1 coverage --include-negatives  # confirms full coverage

# Render the full corpus to training-ready format:
uv run python -m teacher_model.stage1 render --out data/stage1_train.jsonl

# After training, evaluate against the held-out set:
uv run python -m teacher_model.stage1 harness --endpoint http://vllm:8000 --holdout data/holdout_briefings.jsonl
```

Outputs: a `stage1_train.jsonl` with ~2K rendered examples ready for LoRA SFT, plus a `harness_report.md` showing all 7 acceptance metrics with bootstrapped CIs.

## Design

### Approach (locked decisions from /brainstorm 2026-05-08)

1. **Chat-template-native Anthropic-shape output.** Examples are rendered via `tokenizer.apply_chat_template(messages, tools=[...])` against the Stage 0-pinned Qwen3.6-A3B tokenizer. The model emits tool calls through the chat template's tool-call slot; the vLLM/SGLang `qwen3_coder` parser at serving time returns structured `tool_calls[]` to `teacher.ts` with no translation shim.
2. **Emission-only single assistant turn.** Each example is one assistant turn that emits either `[text + tool_use(s)]` or `[text-only]`. No `tool_result` continuations; that capability belongs to Stage 2. Gated by Stage 0 `continuation_degeneracy_rate < 30%` (if exceeded, Q4 contingency adds ~15% organic-narration negatives).
3. **Mixed shape: ~60% synthesis, ~40% chat.** Synthesis-shape examples use real cached briefings from `model/data/eval/inference_cache/auto-t5_http/` rendered through the Python port of `buildSynthesisFraming` and end with the `<analysis>...</analysis>` reasoning scratchpad followed by the response (matching `prompts.ts:158`). Chat-shape examples use a parameterized scenario template producing 1-5 turn conversation histories.
4. **Prose-only negatives** (no restraint narration) across 6 categories: chitchat (15%), premature/listening (25%), ambiguous request (15%), already-recommended (10%), out-of-scope/emotional (15%), borderline-wrong-tool (20%). Includes ~30-50 matched-contrast pairs (positive/negative twins sharing student profile + piece + shape).
5. **Multi-tool turns natively allowed: ~70% single-tool, ~30% multi-tool.** Whitelist of 6 realistic independent-emission combos (e.g. `create_exercise` + `score_highlight`); explicitly forbids dependent combos that need a `tool_result` between calls (those belong to Stage 2). Multi-tool examples carry a `_metadata.combo_rationale` line for review.
6. **Hybrid authoring.** ~70% positives via Sonnet distillation through `UNIFIED_TEACHER_SYSTEM` with the 6-tool palette, validated against the Pydantic mirror of `tool-processor.ts` Zod schemas. ~30% (all negatives + all matched pairs + ~50 adversarial positives) hand-authored as JSON files under `negatives/` and `matched_pairs/`.
7. **Stage 0 prerequisite outputs consumed:** `tokenizer_pin.json` (chat-template integrity), per-category over-call breakdown (gates Q4 negative-format choice), `continuation_degeneracy_rate` (gates Q2 emission-only choice).

### Acceptance criteria (gates "Stage 1 done")

Held-out 12% of authored data (~240 examples), stratified by shape x tool x pos/neg x single/multi-tool x negative-category. Held-out briefings are excluded from Sonnet distillation (source-isolation rule); held-out negatives use briefing-templates not used in train negatives. Frozen at training kickoff -- no iterative test-set tuning.

| Metric | Threshold | How measured |
|---|---|---|
| Serving-runtime parse rate | >=99% | Output extracted via vLLM `qwen3_coder` parser into structured `tool_calls[]`; rate = successful parses / total positive emissions |
| Tool selection accuracy | >=95% | Among held-out positives, correct tool name selected; per-tool breakdown |
| Argument Pydantic validity | >=99% | Each `tool_use.input` round-trips through the Pydantic mirror of `tool-processor.ts` schemas |
| Argument semantic accuracy | >=90% | Manual spot-check on 50 sampled positives: correct enum, plausible bar range, valid slug format, dimension matches signal |
| Negative discrimination | >=90% overall, >=85% on premature/listening | No tool emitted on held-out negatives; per-category breakdown |
| Multi-tool emission distribution | within +-5pp of training | Held-out multi-tool emission rate matches authored ~30% |
| Matched-contrast pair discrimination | >=80% of pairs | For each (pos, neg) twin: model emits tool on pos AND not on neg; per-pair, not per-example |

All gates must pass for Stage 1 to be considered done. Failure on any one triggers diagnosis + remediation (extra examples in deficient cells, hyperparameter tweak, or shape revision) before unfreezing the adapter for Stage 2.

### Why this approach over alternatives

- **Why chat-template-native vs thin Qwen-envelope adapter?** Chat-template-native is the only path that genuinely deletes parsing code -- the serving runtime returns structured `tool_calls[]`, no `<tool_call>` regex extraction. Under the carry-forward LoRA constraint this is a one-way door, so the cleanest target wins.
- **Why emission-only over multi-turn examples?** Stage 1 tests *when* and *how* to call; Stage 2 tests *what to say after results come back*. Carry-forward LoRA means contamination from a low-quality continuation pattern in Stage 1 is sediment Stage 2 has to dredge. Stage 0's continuation probe gates this choice empirically.
- **Why 6-category negative split vs uniform restraint?** Restraint isn't one behavior. Stage 1 has to learn 6 distinct don't-call discriminations independently; per-category coverage prevents the model from learning "tools fire in chat, restraint fires in synthesis" or any other shape-conflated heuristic.
- **Why hand-author negatives vs filter-source from Sonnet?** Sonnet's over-call tendency (project memory: ASCF outcome 1.387) makes filter-sourcing unreliable -- filtering would produce too few clean negatives. Hand-authoring 600 examples against the category breakdown is faster than running enough Sonnet passes to get 600 clean negatives.
- **Why harness through real serving stack (vLLM) rather than custom regex?** The "serving-runtime parse rate" metric is the single most likely silent-failure mode. If the harness uses a custom extractor and prod uses `qwen3_coder`, the harness can pass on shape that prod fails to parse -- the model ships broken.

### Trade-offs chosen

- **Tooling first, corpus later.** This plan ships verifiable infrastructure (~9 modules, ~15 vertical-slice tests) that is TDD-able. Hand-authored corpus generation is execution work (~6-8 weeks of authoring), not TDD work. Decoupling means authoring can start as soon as tooling lands without blocking on corpus completion for plan sign-off.
- **Pydantic mirror over generated bindings.** We hand-port the Zod schemas in `tool-processor.ts` to Pydantic models in `schema.py`. Adding a Zod->Pydantic codegen pipeline is more code than the 6 tools justify; mirror drift is caught by a contract test that fails CI when the TS schemas change without the Python mirror updating.
- **Single tokenizer pin** committed to the repo. Trades flexibility (can't easily swap tokenizers) for reproducibility. Tokenizer upgrades require Stage 0 re-run.
- **Sonnet for distillation despite Stage 4 plan to move past Sonnet voice.** Distillation here trusts only Sonnet's tool-call *mechanics* (correct shape, valid args), not Sonnet's prose. This narrow trust is fine for Stage 1; Stage 2/3 will trust less.

## Modules

- **schema** (`apps/evals/teacher_model/stage1/schema.py`)
  - Interface: `Stage1Example`, `Stage1Negative`, `MatchedContrastPair`, `validate_tool_input(name: str, input: dict) -> list[str]`
  - Hides: Pydantic models for examples and tool calls; the per-tool input validators that mirror the Zod schemas in `tool-processor.ts`; serialization to/from JSONL; the `_metadata` envelope (combo_rationale, contrast_id, source)
  - Tested through: round-trip serialize/parse on fixtures, validator output on known-good and known-bad inputs (one fixture per tool covering each enum value and each conditional-required combination)
- **briefing_source** (`apps/evals/teacher_model/stage1/briefing_source.py`)
  - Interface: `iter_synthesis_briefings(cache_dir: Path) -> Iterator[Briefing]`, `iter_chat_scenarios(template: dict, n: int, seed: int) -> Iterator[Briefing]`
  - Hides: JSON loading from `auto-t5_http/`, the Python port of `buildSynthesisFraming` (or the existing `build_synthesis_user_msg` re-used), the chat scenario template instantiator (parameterized over tool, intent, student profile, piece)
  - Tested through: count + stratification assertions on the 890-briefing pool, deterministic enumeration under fixed seed, port-parity test against a reference TS-generated framing string
- **distill** (`apps/evals/teacher_model/stage1/distill.py`)
  - Interface: `distill(briefing: Briefing, shape: Shape, sonnet: AnthropicClient, system_prompt: str) -> Stage1Example | None`
  - Hides: Anthropic API call with `tool_choice: auto`, content-block extraction (text + tool_use), schema validation via the schema module, retry/backoff on transient errors, `None` return when validation rejects
  - Tested through: stub Anthropic client returning canned responses; assert valid response yields example, invalid yields None, transient error triggers retry then succeeds
- **negatives_loader** (`apps/evals/teacher_model/stage1/negatives_loader.py`)
  - Interface: `load_negatives(dir: Path) -> list[Stage1Negative]`, `load_pairs(dir: Path) -> list[MatchedContrastPair]`
  - Hides: directory enumeration, per-file schema validation, contrast-id cross-reference check (every pair has both members present and same `_metadata.contrast_id`)
  - Tested through: fixture directory with valid + invalid negatives + pair files; assert valid load and clear error messages on each invalid case
- **coverage** (`apps/evals/teacher_model/stage1/coverage.py`)
  - Interface: `CoverageMatrix(targets: dict)`, `record(example: Stage1Example) -> None`, `unfilled_cells() -> list[Cell]`, `is_satisfied() -> bool`, `report() -> str`
  - Hides: per-tool/per-arg-cell key derivation, threshold table (each enum value >=5x, each optional present/absent >=10x each, each conditional-required combo >=5x), shape-stratification, negative-category counts
  - Tested through: feed example fixtures, assert `unfilled_cells()` returns expected cells, `is_satisfied()` flips at exact threshold boundary
- **holdout** (`apps/evals/teacher_model/stage1/holdout.py`)
  - Interface: `split_holdout(briefings: list[Briefing], frac: float, strata: list[str], seed: int) -> tuple[list[BriefingId], list[BriefingId]]`, `commit_manifest(holdout_ids: list[BriefingId], path: Path) -> None`
  - Hides: stratum-balance correction when a stratum is under-populated, deterministic seeded shuffle, source-isolation manifest write
  - Tested through: split a fixture pool, assert proportions match `frac` per stratum within +-1 example, assert determinism across two invocations with same seed, assert different seeds yield different splits
- **render** (`apps/evals/teacher_model/stage1/render.py`)
  - Interface: `render(example: Stage1Example, tokenizer, tools: list[dict]) -> str`, `verify_tokenizer_pin(tokenizer_dir: Path, pin_path: Path) -> None`
  - Hides: tokenizer load, hash verification against Stage 0 pin (raises `TokenizerPinMismatchError` on drift), `apply_chat_template(messages, tools=tools, add_generation_prompt=False)` invocation with the right `tools=[...]` argument shape derived from the schema module
  - Tested through: render a known fixture against a stubbed tokenizer that records inputs (assert tools list passed correctly, messages structure correct), pin verification raises on hash mismatch
- **harness** (`apps/evals/teacher_model/stage1/harness.py`)
  - Interface: `run_harness(endpoint: str, holdout_path: Path, tokenizer_pin: Path) -> HarnessReport`, `HarnessReport.write_markdown(path: Path) -> None`
  - Hides: vLLM client (OpenAI-compatible API), `qwen3_coder` parser invocation (relies on vLLM-side parsing), the 7 metric computations, bootstrapped 95% CIs, per-tool and per-negative-category breakdowns, serving-stack parse-rate measurement, dossier rendering
  - Tested through: stub vLLM endpoint returning canned responses for a small holdout; assert report contains all 7 metrics with values matching expected from the canned outputs
- **cli** (`apps/evals/teacher_model/stage1/cli.py`)
  - Interface: `main()` -- argparse subcommands `holdout`, `distill`, `coverage`, `render`, `harness`
  - Hides: argument parsing, output path resolution, environment-var loading for OpenRouter/Anthropic keys
  - Depth: SHALLOW (justified -- pure orchestration glue; each subcommand is a one-liner that calls into a deep module)
  - Tested through: invoke each subcommand with `--help`, assert exit 0 and expected text; integration test invoking `holdout` end-to-end on a 20-briefing fixture
- **negatives data directory** (`apps/evals/teacher_model/stage1/negatives/` and `matched_pairs/`)
  - Interface: directory of JSON files matching the `Stage1Negative` and `MatchedContrastPair` Pydantic models
  - Depth: data artifact (justified -- the data IS the deliverable; loader is the negatives_loader module)
  - Tested through: `negatives_loader` test suite (the loader test IS the data validation)

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/evals/teacher_model/stage1/__init__.py` | Package marker | New |
| `apps/evals/teacher_model/stage1/schema.py` | Pydantic models + per-tool validators mirroring tool-processor.ts | New |
| `apps/evals/teacher_model/stage1/briefing_source.py` | Synthesis briefing loader + chat scenario generator + Python framing port (or reuse existing) | New |
| `apps/evals/teacher_model/stage1/distill.py` | Sonnet distillation driver | New |
| `apps/evals/teacher_model/stage1/negatives_loader.py` | Hand-authored negatives + matched-pair loader | New |
| `apps/evals/teacher_model/stage1/coverage.py` | Argument-coverage matrix tracker | New |
| `apps/evals/teacher_model/stage1/holdout.py` | Stratified held-out split with source-isolation | New |
| `apps/evals/teacher_model/stage1/render.py` | apply_chat_template renderer with tokenizer pin verification | New |
| `apps/evals/teacher_model/stage1/harness.py` | vLLM-backed harness with 7 metrics | New |
| `apps/evals/teacher_model/stage1/cli.py` | argparse front-end for all subcommands | New |
| `apps/evals/teacher_model/stage1/negatives/.gitkeep` | Directory placeholder for hand-authored negatives | New |
| `apps/evals/teacher_model/stage1/matched_pairs/.gitkeep` | Directory placeholder for matched-contrast pairs | New |
| `apps/evals/teacher_model/stage1/data/holdout_briefings.jsonl` | Source-isolation manifest (generated then committed) | New |
| `apps/evals/teacher_model/stage1/data/coverage_targets.json` | Per-tool argument-coverage thresholds (the matrix definition) | New |
| `apps/evals/teacher_model/stage1/data/chat_scenario_template.json` | Parameterized chat scenario template for `iter_chat_scenarios` | New |
| `apps/evals/teacher_model/stage1/tests/test_schema.py` | Validator coverage per tool | New |
| `apps/evals/teacher_model/stage1/tests/test_briefing_source.py` | Source counts + stratification + framing port-parity | New |
| `apps/evals/teacher_model/stage1/tests/test_distill.py` | Anthropic stub + retry + validation rejection | New |
| `apps/evals/teacher_model/stage1/tests/test_negatives_loader.py` | Valid load + invalid file error reporting | New |
| `apps/evals/teacher_model/stage1/tests/test_coverage.py` | Cell key derivation + boundary satisfaction | New |
| `apps/evals/teacher_model/stage1/tests/test_holdout.py` | Stratification + determinism + source-isolation | New |
| `apps/evals/teacher_model/stage1/tests/test_render.py` | Pin verification + apply_chat_template invocation | New |
| `apps/evals/teacher_model/stage1/tests/test_harness.py` | Stub vLLM + report contents | New |
| `apps/evals/teacher_model/stage1/tests/test_cli.py` | Subcommand smoke tests | New |
| `apps/evals/teacher_model/stage1/tests/test_schema_contract.py` | Contract test: Pydantic mirror parity vs tool-processor.ts Zod schemas | New |
| `apps/evals/teacher_model/tool_format.py` | Add deprecation docstring noting Stage 1 obsoletes this module | Modify |

## Open Questions

- **Q:** Should the Python port of `buildSynthesisFraming` be a fresh write or reuse the existing port in `apps/evals/teaching_knowledge/run_eval.py` (`build_synthesis_user_msg`)? **Default:** reuse the existing port; add a port-parity test in `test_briefing_source.py` that compares its output against a TS-generated reference for 5 fixture inputs. Re-port from scratch only if the existing port lacks coverage of `getStyleGuidance` or the voice-blocks pipeline.
- **Q:** Source for chat-shape `dynamicContext` strings (~800 examples worth)? **Default:** parameterized template generator producing realistic strings (`Student level: intermediate. Goals: ...`). If production `chatV6` traces accumulate during pre-beta, fold real ones in as highest-priority seeds (replacing template output for those scenarios).
- **Q:** vLLM vs SGLang for the harness serving stack? **Default:** vLLM with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. SGLang is the alternative if vLLM's qwen3_coder parser proves unstable on multi-tool emissions; harness module isolates the client behind an `EvalClient` interface so swap is contained.
- **Q:** Does the system prompt the synthesis-shape examples train against use `UNIFIED_TEACHER_SYSTEM` (production teacher.ts) or `apps/shared/teacher-style/synthesis_system.txt` (eval pipeline)? **Default:** `UNIFIED_TEACHER_SYSTEM` -- that is what production uses. The cached briefings' user-message content is system-prompt-agnostic. Note this divergence from the Stage 0 spec's choice (which used the eval-path prompt) and verify briefings make sense under either.
- **Q:** What happens if the production tool palette in `tool-processor.ts` changes between this spec and Stage 1 training? **Default:** the contract test (`test_schema_contract.py`) fails CI; Stage 1 corpus generation pauses until the Pydantic mirror is updated and existing examples are re-validated. Examples that no longer validate are quarantined and either fixed or replaced.
