# Audio-Native Teacher Pivot (Wave 1) Design

**Goal:** Stand up the gated audio-native-teacher program — epic + Gate 0 issue + contingency issue + old-plan migration — and ship an offline-testable Gate 0 probe harness (`model/src/audio_teacher/`) that the user can later run against Tinker/Inkling under a $50 hard cap.
**Not in scope:** Gate 1+ training runs or issues, the contrastive rendering engine, grounding/verifier code (#101/#67/#108 lanes), pedagogy training, the frozen-encoder Bradley-Terry comparator baseline, any production deploy, and actually RUNNING the probe against Tinker (the funded run is executed by the user afterward on the Gate 0 issue).

## Problem

The teacher-model roadmap is split across two now-obsolete issue sets: the Qwen 5-stage teacher-finetune (`epic:teacher-finetune`: #16 #55, plus deferred #32/#33/#40) and the MuQ/Aria encoder-v2 training epic (`epic:model-v2`: #71 #79 #80 #81 #82 #83 #84). Both assume a text-only teacher fed by separately-graded encoder streams. The Inkling/Tinker era makes a different architecture testable cheaply: one audio-native LoRA-tuned open multimodal model that collapses ear+teacher. Nothing in the repo can currently (a) record that decision durably, (b) probe whether base Inkling can even hear piano-performance contrasts before money is spent, or (c) prevent the two dead issue sets from being mistaken for the live plan.

## Solution (from the user's perspective)

After this ships, the user has:
1. A GitHub epic (`epic:audio-teacher`) that is the single durable roadmap: architecture decision, 4-layer data plan, gate ladder with ex-ante kill criteria, wave-2 issue drafts as a checklist, and standing policies.
2. A Gate 0 issue with kill criteria fixed in the body before any run.
3. A parked contingency issue (DIY audio-native path) that activates only on a failed gate — old plans never resurrect.
4. Nine dead issues closed ("not planned", successor linked); #32/#33/#40 relabeled to `epic:audio-teacher` with retarget comments.
5. `cd model && uv run python -m audio_teacher.probe --manifest M.yaml ...` — a harness that loads a contrast-pair manifest, refuses to run on missing/malformed/offloaded audio (loud errors with the exact rehydrate command), elicits forced A/B judgments per contrast axis, enforces the $50 cap *before* the overshooting call, resumes from saved responses, and emits a deterministic JSON report with per-axis × per-population accuracy and a PASS/FAIL verdict. All tests run offline; no test ever calls Tinker.

## Design

- **Two deliverable kinds, one branch.** GitHub-graph mutation (epic, Gate 0, contingency, migration) is command-work verified by before/after `gh issue list` snapshots in ship notes — never code-tested. The harness is TDD'd Python in the uv-managed `model/` package.
- **Harness follows the `follower_bench` pattern:** flat modules under `model/src/audio_teacher/`, tests under `model/tests/audio_teacher/`, argparse CLI drivers run as `python -m audio_teacher.<mod>`, `__file__`-anchored default paths (never CWD-relative), manifest-in / JSON-report-out. `src/audio_teacher` is added to the hatch wheel `packages` list (required for imports under pytest — same as `follower_bench`).
- **Offload awareness is self-contained.** `model/src/paths.py` is not importable from installed packages (it is not in the wheel `packages` list; `follower_bench` self-anchors for the same reason). The manifest loader re-implements the ~30-line `ensure_local` rehydrate-hint logic against `data/manifests/r2_offload.json`, with an injectable registry path so the behavior is testable. T2 raw audio (`data/raw/competition`) is R2-offloaded today, so this path will be hit on the first real curation.
- **Tinker behind a Protocol.** `ProbeClient` protocol (`estimate_cost_usd(pair)`, `ask(pair)`); `RecordedResponseClient` (JSONL replay, used by every test and offline re-scoring) and `TinkerProbeClient` (real; lazy-imports `tinker` + `tinker_cookbook` + `tml_renderers`, builds chat messages with `chat.AudioPointer` per the Inkling audio cookbook, raises `TinkerNotInstalledError` with install instructions if the SDK is absent). Tests only ever touch the recorded fake and the not-installed error path.
- **Budget guard is a hard pre-call gate.** `BudgetGuard.precheck(estimated_cost)` raises `BudgetExceededError` BEFORE the call that would exceed `--max-spend` (default 50.0). No warning mode exists. USD-per-1M-token rates are required CLI arguments in live mode (no silent zero-rate default that never trips).
- **Scoring is population-partitioned by construction.** Report cells are keyed `axis/population` (`pedaling/real`, `pedaling/synthetic`, ...); no pooled number exists anywhere in the report (the #21 synthetic-gap scar). The gate verdict reads ONLY real-population cells; synthetic cells are informative. Ex-ante constants: accuracy >= 0.70 per axis on real pairs, >= 20 real pairs per axis, unparseable-response rate <= 0.10. Any violation, including insufficient or ambiguous data, yields FAIL — uncertain defaults to closed.
- **Failure posture:** explicit exceptions everywhere; Tinker API errors propagate as-is; responses are appended to `responses.jsonl` per call, so a re-run resumes from the manifest (already-answered pairs are skipped); the report is only written when every manifest pair has a response.
- **Manifest builder stays shallow by design** (Gate 0 is a probe, not a rendering framework): CSV rows -> YAML manifest, then round-trips through the real loader so curation output is loadable by construction.
- **Reference drift handled:** the approved design cited follower #119 for Layer-2 grounding; #119 is CLOSED (HMM shipped). The epic body cites the follower lane as #108 (epic) / #126 (timing-aware HMM) with #119 noted as shipped.

## Modules

**`audio_teacher.manifest`** (DEEP)
- Interface: `load_manifest(manifest_path, repo_root=..., offload_registry=None) -> ProbeManifest`; dataclasses `ProbeManifest`, `ContrastPair`; `ManifestError`; constants `AXES`, `POPULATIONS`.
- Hides: YAML schema validation (v1: `schema_version`, `sample_rate`, `pairs[]` with `id/axis/population/clip_a/clip_b/degraded/description`), duplicate-id detection, R2-offload registry lookup with exact rclone/regen rehydrate command in the error, per-clip WAV validation fan-out. Guarantees: a returned manifest has every clip present and valid — a probe over a biased subsample is impossible.
- Tested through: loading fixture manifests (valid; missing clip + temp offload registry -> error contains `rclone copy`; schema violations -> `ManifestError`).

**`audio_teacher.audio`** (DEEP)
- Interface: `validate_wav(path, expected_sample_rate) -> WavInfo`; `MalformedClipError`.
- Hides: RIFF/WAVE parsing via stdlib `wave`, mono/rate/zero-length checks, truncation detection (declared frames vs actual payload bytes).
- Tested through: generated fixtures — valid mono clip passes; stereo / wrong-rate / truncated each abort with the file named in the message.

**`audio_teacher.prompts`** (DEEP — owns the elicitation contract end to end)
- Interface: `build_question(axis) -> str`; `parse_choice(text) -> "a" | "b" | None`.
- Hides: per-axis degradation phrasing (pedaling/dynamics/phrasing), the strict `ANSWER: A|B` answer-format instruction, tolerant last-answer-wins parsing.
- Tested through: question text carries the axis contrast + answer instruction; canned completions parse to a/b/None.

**`audio_teacher.client`** (DEEP)
- Interface: `ProbeClient` Protocol; `ProbeResponse`; `RecordedResponseClient(responses_path)`.
- Hides: JSONL replay bookkeeping; loudly `KeyError`s on a pair with no recorded response (incomplete fixture).
- Tested through: replaying canned JSONL; missing-pair error.

**`audio_teacher.tinker_client`** (DEEP; never imported by tests except its not-installed error path)
- Interface: `TinkerProbeClient(sample_rate, usd_per_1m_input_tokens, usd_per_1m_output_tokens, max_tokens)`; `TinkerNotInstalledError`; `INKLING_MODEL`.
- Hides: tinker/tml_renderers wiring — renderer construction, `AudioPointer` message assembly for clip A/B, sampling call, response decode, token-count-based cost accounting, duration-based pre-call cost estimation.
- Tested through: constructing without the SDK installed raises `TinkerNotInstalledError` naming the install command (skipped if the SDK is present).

**`audio_teacher.budget`** (DEEP)
- Interface: `BudgetGuard(max_spend_usd)`, `.precheck(estimated_next_cost_usd)`, `.record(actual_cost_usd)`, `.spent_usd`; `BudgetExceededError`.
- Hides: projection arithmetic and the raise-before-call invariant.
- Tested through: simulated cost accumulation — the overshooting precheck raises and spend is unchanged.

**`audio_teacher.scorer`** (DEEP)
- Interface: `score_responses(manifest, responses: Mapping[pair_id, text]) -> dict`; `render_report(report) -> str`; `ProbeIncompleteError`; constants `KILL_THRESHOLD=0.70`, `MIN_REAL_PAIRS_PER_AXIS=20`, `MAX_UNPARSEABLE_RATE=0.10`.
- Hides: choice-vs-degraded correctness, per-`axis/population` cell aggregation, verdict logic (real-only gating, ambiguity -> FAIL with reasons), deterministic serialization (sorted keys; no timestamps in the report — volatile metadata lives in a separate `run_meta.json`).
- Tested through: in-memory manifests + canned responses -> exact cell values, PASS/FAIL verdicts per ex-ante rules, byte-identical re-render.

**`audio_teacher.probe`** (DEEP — the CLI composition root)
- Interface: `main(argv) -> int`; CLI `--manifest --run-dir --recorded --max-spend --usd-per-1m-input-tokens --usd-per-1m-output-tokens --max-tokens`.
- Hides: client selection (recorded vs Tinker), resume-from-`responses.jsonl`, per-call budget precheck/record, report + run-meta writing, exit code from verdict.
- Tested through: offline end-to-end run over a fixture manifest with `--recorded` (report written, exit code matches verdict); resume run proves answered pairs are never re-asked.

**`audio_teacher.build_manifest`** (SHALLOW — deliberate; Gate 0 needs curation, not a rendering framework. It is a thin CSV->YAML adapter whose validation is delegated to `load_manifest`.)
- Interface: `main(argv) -> int`; CLI `--pairs-csv --sample-rate --out`.
- Tested through: CSV + real clips -> YAML that round-trips through `load_manifest`; invalid axis rejected before writing.

## Verification Architecture

- Canonical success state: (1) `uv run python -m pytest tests/audio_teacher -v` green in `model/`; (2) the offline CLI run over the committed fixture path (`python -m audio_teacher.probe --manifest <tmp fixture> --recorded <canned jsonl>`) writes `report.json` with population-partitioned cells and a verdict; (3) `gh issue list` shows: epic + Gate 0 + contingency open under `epic:audio-teacher`, the nine dead issues closed, #32/#33/#40 relabeled.
- Automated check: the pytest suite (offline; no network). The determinism test doubles as the golden check: same manifest + same canned responses -> byte-identical `render_report` output.
- Harness: no separate Task Group 0 — the test suite is built vertically alongside each module (every task is test-first), and the migration half is command-work verified by `gh issue list --json` snapshots captured in the ship notes, per the approved design ("Migration is NOT code-tested").

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/pyproject.toml` | add `src/audio_teacher` to hatch wheel `packages` | Modify |
| `model/src/audio_teacher/__init__.py` | package marker + one-line docstring | New |
| `model/src/audio_teacher/audio.py` | WAV header validation | New |
| `model/src/audio_teacher/manifest.py` | YAML schema + loader + offload rehydrate hints | New |
| `model/src/audio_teacher/prompts.py` | elicitation questions + answer parsing | New |
| `model/src/audio_teacher/client.py` | ProbeClient protocol + ProbeResponse + recorded fake | New |
| `model/src/audio_teacher/tinker_client.py` | real Tinker/Inkling client (lazy SDK import) | New |
| `model/src/audio_teacher/budget.py` | hard pre-call spend cap | New |
| `model/src/audio_teacher/scorer.py` | per-cell scoring, verdict, deterministic report | New |
| `model/src/audio_teacher/probe.py` | CLI driver: manifest -> responses -> report | New |
| `model/src/audio_teacher/build_manifest.py` | CSV -> YAML curation script | New |
| `model/tests/audio_teacher/__init__.py` | test package marker | New |
| `model/tests/audio_teacher/conftest.py` | shared WAV/manifest fixture builders | New |
| `model/tests/audio_teacher/test_audio.py` | WAV validation behavior | New |
| `model/tests/audio_teacher/test_manifest.py` | loader behavior incl. rehydrate hint | New |
| `model/tests/audio_teacher/test_prompts.py` | question build + choice parsing | New |
| `model/tests/audio_teacher/test_client.py` | recorded client replay + missing-pair error | New |
| `model/tests/audio_teacher/test_tinker_client.py` | not-installed error path | New |
| `model/tests/audio_teacher/test_budget.py` | cap raises before overshoot | New |
| `model/tests/audio_teacher/test_scorer.py` | cells, verdicts, determinism | New |
| `model/tests/audio_teacher/test_probe.py` | offline e2e + resume | New |
| `model/tests/audio_teacher/test_build_manifest.py` | CSV->YAML round-trip | New |
| GitHub: label `epic:audio-teacher`; epic, Gate 0, contingency issues; close #71 #79 #80 #81 #82 #83 #84 #16 #55; relabel #32 #33 #40 | issue-graph migration | Command-work |

## Open Questions

- Q: Exact PyPI package names / versions for the Tinker SDK trio (`tinker`, `tinker-cookbook`, `tml-renderers`) at probe-run time. Default: they are NOT added to `model/pyproject.toml`; `TinkerProbeClient` lazy-imports and raises `TinkerNotInstalledError` with `uv add tinker tinker-cookbook` guidance. The user verifies names when funding the Gate 0 run (noted in the Gate 0 issue body).
- Q: Inkling's audio token rate (DMel tokens/second) for pre-call cost estimation. Default: conservative constant `AUDIO_TOKENS_PER_SECOND_ESTIMATE = 100` in `tinker_client.py`; overestimating only trips the cap earlier, never later — safe direction. Actual recorded cost uses real token counts from the sampling response.
- Q: Whether the future real T2 curation uses 16 kHz mono re-encodes of `data/raw/competition` audio. Default: manifest `sample_rate` is a required field validated against every clip header, so any choice is enforced explicitly at curation time; the harness does not resample.
