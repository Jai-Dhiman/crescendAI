# Transkun Transcriber Migration Design

**Goal:** Replace the incumbent aria-amt piano transcriber with Transkun (MIT, ISMIR 2024) repo-wide, behind an unchanged HTTP `/transcribe` contract and a single shared helper, so every transcription consumer (live service + offline measurers) gets Transkun's superior offset/velocity/pedal fidelity with zero contract churn.
**Not in scope:** Stage-2 Transkun fine-tuning; the #104 difficulty-head re-fit (closed research line, see MEMORY); any change to what the teacher LLM does with the notes; MuQ audio-chunker (`apps/inference/audio_chunker.py`) which is unrelated (MuQ 24kHz path, has live importers).

## Problem
The transcriber `apps/inference/amt/transcription.py` (`EndpointHandler`) runs EleutherAI aria-amt via a bespoke encoder-decoder + hand-written greedy-decode grammar tracker (`advance_valid_note_groups`), a config monkeypatch (`_patched_load_config`), and a same-pitch re-onset deduper (`deduplicate_notes` / chroma `DEDUP_WINDOW_S`). Aria-amt gets note offsets/durations wrong (offset F1 0.37 vs Transkun 0.79 on MAESTRO, #125 Stage 0), over-transcribes ~4%, and requires two `git+` dependencies plus a hybrid ONNX-encoder/PyTorch-decoder container (`server.py` + `scripts/export_onnx.py`) whose only rationale was aria-amt CPU throughput. Transkun (`pip install transkun`, whole-piece, semi-CRF onset+offset, emits velocity + CC64 pedal) is a strictly better substrate but its torch/deps conflict with `model/.venv` (documented gotcha [[project_uv_run_mutates_model_venv]]), so it cannot simply be imported into the offline measurers.

Consumers that must keep working unchanged through the swap:
- `apps/api/src/services/inference.ts` — POSTs `{chunk_audio[, context_audio]}` to `${AMT_ENDPOINT}/transcribe`, reads `midi_notes` / `pedal_events`, treats an error-shaped body as `InferenceError` → Tier-3 degrade.
- `model/src/chroma_dtw_eval/amt_regen.py` — POSTs 27s chunks to the same `/transcribe`, builds pseudo-truth.
- `model/src/score_library/pieceid_amt_axis.py` — POSTs 15s chunk + context to `/transcribe`.
- Eight offline measurer scripts import `from transcription import EndpointHandler` and call `handler._transcribe(pcm)` in-process against `model/.venv`.

## Solution (from the user's perspective)
Nothing user-facing changes. The live web/iOS practice loop still POSTs audio chunks and receives `midi_notes` + `pedal_events`; internally those notes now come from Transkun. Offline eval and measurement pipelines produce the same bundle/response shapes with a better substrate. Transcription failures become loud (`TranskunError`, service refuses to start if no Transkun path resolves) instead of silently returning empty notes.

## Design
**One shared deep module `apps/inference/amt/transkun_cli.py`** that does NOT import `transkun`; it shells out via `uv run --no-project --with transkun --python 3.11 transkun IN.wav OUT.mid --device cpu` (verified CLI signature; default device is already `cpu`; `--no-project` is mandatory or a bare `uv run --with` from `model/` rebuilds the shared `.venv`). Because it only ever subprocesses, it is import-safe from BOTH `model/.venv` (measurers) and the service env. It parses `OUT.mid` with `pretty_midi` (notes carry velocity; `control_changes` number 64 → pedal events, value ≥ 64 = "on" → 127). Public interface:
- `transcribe_wav(wav_path) -> (notes, pedals)`
- `transcribe_pcm(pcm_16k) -> (notes, pedals)` — writes a temp 16k-mono WAV, delegates to `transcribe_wav`.
- `midi_to_notes_and_pedals(midi_path) -> (notes, pedals)` — the pretty_midi parse (velocity + CC64 semantics), the correctness crux, tested in isolation.
- `TranskunError(RuntimeError)` — raised on non-zero subprocess exit, missing output MIDI, or missing input; NEVER return empty notes on failure.

Returned dict shapes are exactly what both surfaces already expect: `{"pitch": int, "onset": float, "offset": float, "velocity": int}` and `{"time": float, "value": int}`.

**Surface 1 — HTTP service** (`transcription.py`, `amt_local_server.py`, `server.py`): the `/transcribe` contract is UNCHANGED (accepts `chunk_audio` [+ optional `context_audio`], returns `midi_notes` + `pedal_events` + `transcription_info`), so `inference.ts`, `amt_regen.py`, and `pieceid_amt_axis.py` need ZERO changes. `context_audio` becomes accepted-but-ignored (aria overlap/dedup semantics deleted). `EndpointHandler` collapses to: WebM→PCM (KEEP the existing ffmpeg decode) → transcribe → same response shape. `__init__` resolves a transcriber ONCE (prefer a warm in-process `transkun` Python API if importable in the service env; else fall back to the `transkun_cli` CLI helper; if NEITHER is available, raise and refuse to start — never a silent per-request fallback). The container `server.py` drops the ONNX-encoder/PyTorch-decoder split and its build machinery.

**Surface 2 — offline measurers** (`model/.venv`): replace `from transcription import EndpointHandler; handler._transcribe(pcm)` with `from transkun_cli import transcribe_pcm; transcribe_pcm(pcm)` across `gd_rate/transcribe_bundles.py`, `ga_validation/*`, `gc_error_bars/*`, `tau_calibration/*`, `amt_fidelity/*`, `dynamics_supply/*`.

**Chroma retune** (`amt_regen.py`): drop the Aria same-pitch re-onset deduper (`DEDUP_WINDOW_S` → pass-through) because Transkun does not emit that artifact; re-check the `MIN_ANCHORS`/`MIN_SPAN_FRACTION`/`MAX_ANCHOR_GAP_S` acceptance gates against Transkun's ~7% UNDER-transcription — but only adjust thresholds if Gate 1 (chroma recall) regresses, with the delta recorded in #128 (thresholds cannot be re-derived without running the gate).

**Deletions:** aria's two `git+` deps, `_patched_load_config`, `deduplicate_notes`, `advance_valid_note_groups`, all `context_audio` concatenation plumbing, `scripts/export_onnx.py` + the Dockerfile ONNX build stage + `onnx`/`onnxruntime` container deps.

**Trade-offs chosen:**
- Shell-out over in-process import for the shared helper: pays a subprocess + weight-reload cost per measurer call, bought env-isolation that is otherwise impossible (torch conflict). The service recovers the warm-model cost via the optional in-process path.
- Keep the HTTP contract byte-for-byte: avoids touching `inference.ts`/`amt_regen.py`/`pieceid_amt_axis.py` and their Tier-3 degrade wiring, at the cost of carrying an ignored `context_audio` field.
- Two design simplifications are DEFERRED (recorded as follow-ups on #128), NOT silently assumed: (a) collapsing the measurers' 27s stratified windowing to whole-piece transcription — changes measurement semantics, must be proven number-neutral first; (b) unifying dev `amt_local_server.py` with container `server.py` now the ONNX rationale is gone — must be proven behavior-neutral first. First slice keeps existing windowing/structure and only swaps the transcription call.

## Corrections to the approved design (found while reading the code)
- `apps/inference/audio_chunker.py` is NOT importer-less: it is imported by `apps/evals/load_test.py`, `apps/evals/model/skill_eval/run_inference.py`, `apps/evals/inference/eval_runner.py`, and `apps/inference/muq_chunk_compare.py`. It is the MuQ 24kHz chunker, unrelated to AMT. It is NOT deleted.
- `src/export_onnx.py` does not exist; only `apps/inference/amt/scripts/export_onnx.py` exists. Only that file (plus the Dockerfile ONNX stage) is deleted.
- `substrate_versions` strings are provenance metadata only; the `claim_taxonomy` measurer tests use their own `{"amt": ...}` fixtures and do NOT assert the producer string, so changing `"aria-amt/piano-medium-double-1.0"` → `"transkun/2.0.1"` does not break Gate 3.

## Modules
- **`transkun_cli`** (`apps/inference/amt/transkun_cli.py`, new)
  - Interface: `transcribe_wav`, `transcribe_pcm`, `midi_to_notes_and_pedals`, `TranskunError`.
  - Hides: temp-WAV writing, the exact `uv run --no-project --with transkun` invocation, CPU forcing, pretty_midi traversal + CC64 semantics, sort order, loud error handling.
  - Tested through: `midi_to_notes_and_pedals` on a constructed fixture MIDI (velocity + CC64); `transcribe_wav` missing-input raises `TranskunError`; `transcribe_pcm` end-to-end on the real sample WAV (gated on the model).
  - Depth: DEEP — a 3-function surface hiding subprocess orchestration + MIDI parsing.
- **`transcription.EndpointHandler` / `build_response` / `resolve_transcriber`** (`apps/inference/amt/transcription.py`, rewritten)
  - Interface: `EndpointHandler(path).__call__(data) -> response`; `build_response(notes, pedals, chunk_duration_s, elapsed_ms)`; `resolve_transcriber()`.
  - Hides: WebM→PCM decode, transcriber resolution (warm-vs-CLI, refuse-to-start), response assembly.
  - Tested through: `build_response` shape; `resolve_transcriber` refuse-to-start; `__call__` missing-chunk error path (fast); full happy path via Gate 4 smoke.
  - Depth: DEEP.

## Verification Architecture
- **Canonical success state:** every consumer of `/transcribe` and `transkun_cli` receives correctly-shaped notes (with velocity) + pedal events sourced from Transkun; the five validation gates pass; no `EndpointHandler`/aria-amt import remains in swapped consumers.
- **Automated check (fast, model-free):** `pytest apps/inference/amt/test_transkun_cli.py apps/inference/amt/test_transcription.py` (parse + CC64 + error + response-shape + refuse-to-start) and the model-measurer/claim-taxonomy suites.
- **Harness (Task Group 0):** the `transkun_cli` deep module IS the harness — buildable and testable before any surface is rewired, via a pretty_midi fixture MIDI (no model needed for the parse tests). The one real-model test (`transcribe_pcm` on the sample WAV) doubles as Gate 4.
- **Gated integration (model + rehydrated data, run by build/ship):** Gate 1 `just amt-regen-pseudo-truth` + `just chroma-eval-verify` (recall not regressed); Gate 2 `just piece-id-feasibility` (unseen-generator recall stable); Gate 4 `just amt` + `smoke_test_amt.py`; Gate 5 re-baseline evals hard-coding aria numbers, deltas recorded on #128.

## File Changes
| File | Change | Type |
|------|--------|------|
| `apps/inference/amt/transkun_cli.py` | New shared shell-out helper + pretty_midi parse | New |
| `apps/inference/amt/test_transkun_cli.py` | Tests for the helper | New |
| `apps/inference/amt/transcription.py` | Rewrite `EndpointHandler`; add `build_response`, `resolve_transcriber`; delete aria imports, `_patched_load_config`, `deduplicate_notes`, `advance_valid_note_groups`, context concat | Modify |
| `apps/inference/amt/test_transcription.py` | Rewrite: drop deleted-helper tests; add build_response/refuse-to-start/missing-chunk tests | Modify |
| `apps/inference/amt/amt_local_server.py` | `/// deps` (drop aria git deps; add transkun, pretty_midi, soundfile); delegate to new handler; health model "transkun" | Modify |
| `apps/inference/amt/server.py` | Remove ONNX encoder + PyTorch decoder + aria imports; delegate to transkun_cli; keep `/transcribe` + `/health` | Modify |
| `apps/inference/amt/Dockerfile` | Drop ONNX builder stage, `export_onnx.py` copy, onnx/onnxruntime deps; add transkun | Modify |
| `apps/inference/amt/scripts/export_onnx.py` | Delete (ONNX split retired) | Delete |
| `apps/inference/smoke_test_amt.py` | `/// deps` drop aria git dep | Modify |
| `apps/inference/extract_amt_midi.py` | Swap EndpointHandler→transkun_cli; `/// deps` | Modify |
| `model/src/claim_measurement/gd_rate/transcribe_bundles.py` | Swap to transkun_cli; `_transcribe_windows` takes a callable; substrate string; `/// deps` | Modify |
| `model/src/claim_measurement/ga_validation/amt_dynamics_ga_render.py` | Swap EndpointHandler→transkun_cli | Modify |
| `model/src/claim_measurement/ga_validation/amt_dynamics_gb_gate.py` | Swap EndpointHandler→transkun_cli | Modify |
| `model/src/claim_measurement/ga_validation/amt_pedaling_ga_render.py` | Swap EndpointHandler→transkun_cli | Modify |
| `model/src/claim_measurement/gc_error_bars/gc_dynamics_render.py` | Swap EndpointHandler→transkun_cli | Modify |
| `model/src/claim_measurement/tau_calibration/tau_pedaling_render.py` | Swap EndpointHandler→transkun_cli | Modify |
| `model/src/claim_measurement/amt_fidelity/onset_duration_render.py` | Swap EndpointHandler→transkun_cli | Modify |
| `model/src/claim_measurement/dynamics_supply/render_percepiano_bundles.py` | Swap EndpointHandler→transkun_cli; substrate string | Modify |
| `model/src/chroma_dtw_eval/amt_regen.py` | Drop `DEDUP_WINDOW_S` merge (pass-through); MIN_ANCHORS re-check per Gate 1 | Modify |
| `model/config/amt_version.json` | model_name/checkpoint_hash/regen_source_default → transkun/2.0.1 | Modify |
| `docs/apps/07-evaluation.md`, `docs/model/01-data.md`, `docs/model/claim-verifier-signed-d-conventions.md`, `docs/architecture.md`, `CLAUDE.md` | Prose: aria-amt transcriber → Transkun | Modify |
| `Justfile` | Recipe comments/labels aria-amt → transkun (`amt`, `amt-extract`, `amt-run`, `catalog-pieceid-amt-axis`) | Modify |

## Open Questions
- Q: Exact `transkun` version label for `amt_version.json` and `substrate_versions`?  Default: `transkun/2.0.1` (the installed version, probed 2026-07-23); Task 13 re-probes and uses the actually-installed version.
- Q: Do Transkun's ~7% fewer notes push chroma pseudo-truth below the `MIN_ANCHORS=100` / `MAX_ANCHOR_GAP_S=8.0` gate?  Default: keep thresholds; if Gate 1 recall regresses, lower `MIN_ANCHORS` / relax the gap and record the before/after recall on #128.
- Q: Does the service env get `transkun` installed (warm path) or rely solely on the CLI fallback?  Default: install `transkun` in the `amt_local_server.py` `/// deps` so the warm path is exercised; the CLI fallback remains for envs without it.
