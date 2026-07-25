# Implementation Notes — Transkun Transcriber Migration (#128)

Decisions, deviations, and tradeoffs made during build. Read before /review.

## Execution mode
No general-purpose subagent-spawn tool is available in this environment, so the
build controller (this agent) executes each task directly with full /build rigor:
strict TDD red->green, per-task commits, two-stage self-review (spec then quality),
plan-checkbox tracking. Documented here for transparency.

## Key decisions / deviations

- Task 6 (warm in-process path): the plan assumed `transkun.transcribe.transcribe(in, out, device=)`.
  That function does NOT exist in transkun 2.0.1 (only `main()` CLI entry + helpers). A true
  LOAD-ONCE warm path IS achievable via the lower-level API: load the model once
  (moduleconf.parseFromFile + TransKun(conf) + torch.load of the bundled pretrained/2.0.pt),
  then per request write PCM to a temp WAV and feed it through transkun's OWN readAudio + soxr
  resample before model.transcribe. This makes the warm path's audio preprocessing byte-identical
  to the model-verified CLI path (Task 4), amortizing only the model load. Implemented that way.
  The two unit tests monkeypatch `_import_warm_transcriber`, so the warm path is exercised only at
  Gate 4 (just amt + smoke). CLI fallback (`transkun_cli.transcribe_pcm`) remains the guaranteed path.
- Task 11 extract_amt_midi.py: bespoke rewrite (NOT mechanical). WAV inputs now go through
  `transkun_cli.transcribe_wav(path)` directly; sanity fields (pitch_range, ms) reconstructed
  locally from notes + wall-clock; dropped the aria-only `release_accelerator_memory()`/torch cache
  logic (transcribe_wav shells out, no in-process accelerator state).
- Task 11 onset_duration_render.py: aria implicitly truncated to 30s; Transkun does NOT. Added an
  EXPLICIT PCM cap (audio[:AMT_WINDOW_S*SR]) before transcribe_pcm to keep number-neutral. MUST be
  confirmed at Gate 5.
- Task 14 test had an off-by-one (`parents[2]` -> apps/, not repo root); fixed to `parents[3]`.
- Task 15 Dockerfile: single Transkun runtime stage. Supplies BOTH resolve_transcriber paths — warm
  in-process (transkun installed; bundles its own 2.0.pt weights) AND `uv` on PATH for CLI fallback.
- torch.load of transkun's bundled first-party checkpoint keeps weights_only default (matches
  upstream main(); the checkpoint embeds config objects that weights_only=True cannot load).

## Validation gate results (Group F)

- Gate 4 (just amt + smoke): PASS. Smoke on committed fixture = 84 notes, 2 pedals, int
  velocities>0, pitch_range [36,87]. Server /health = {"model":"transkun","loaded":true}.
  Warm in-process path output is BYTE-IDENTICAL to the CLI path (same 84 notes / pitch_range /
  first note pitch=43 onset=0.013 vel=75) and LOAD-ONCE (1 model-load event served the transcribe).
- Gate 3 (measurer suites): PASS. 34 claim-taxonomy (onset/dynamics/pedaling/integration) +
  5 model (pedal_threading, extract_bundle_pedals) = 39 green. No expectation changes needed.
- Gate 1 (chroma-eval-verify): RECALL IMPROVED. Regenerated Transkun pseudo-truth for both
  manifest pieces (bach_prelude_c_wtc1/VID0 = 560 notes/170 anchors; bach_invention_1/7zVlDxBO5q4
  = 475/146). Primary localization recall 40.0 -> 45.0 (+5pp within 1.5s tol). Required the
  anchor gap-gate retune 8.0->9.0 (Transkun's clean 560 notes, no re-onset dedup, shifted anchor
  spacing; one 8.16s gap tripped the old 8.0 cap -- plan Task 18 Step 3 authorized this). The
  dead-reckon-residual-vs-error AUC guard g2 regressed 0.667->0.586 (thresh 0.635 @ n=20), so
  `chroma-eval-verify` exits non-zero on the guard. baseline.json NOT ratcheted (no silent
  down-ratchet). Recall (the parent's criterion) passed; g2 flagged for review.
- Gate 2 (piece-id-feasibility): COULD-NOT-RUN. `just piece-id-feasibility` runs
  `python -m piece_id_eval.cli` which does NOT exist (pre-existing on main; #128 did not touch the
  recipe or piece_id_eval). Real entry is piece_id_eval.bakeoff (synthetic smoke = VERDICT PROCEED,
  harness healthy), but real-mode reads aria-era cached amt_notes; a genuine Transkun-stability
  measurement needs regenerating that notes cache under Transkun from the (large, partly offloaded)
  piece-ID audio corpus -- out of scope here. Reported, not faked.
- Gate 5 (re-baseline aria numbers): onset_duration_render.py re-run under Transkun (N=6) =
  onset noise 3.6ms vs aria 4.37ms, median|err| 2.1ms vs 2.93ms, recall 1.0, VIABLE (<<30ms) --
  NUMBER-NEUTRAL; 2 clips hit [TRUNC] confirming the explicit 30s cap engages cleanly (caution c
  resolved). Full 51-clip aria result restored (spot-check, not a full re-baseline). Completed the
  missed amt_to_json.py swap + amt_regen fallback default + amt_local_server stale comment.
  NOT done (deliberate follow-up): full re-measure of claim_taxonomy.json / pedaling.py perceptual
  numbers (whole_piece ref 0.4623, partial-rho, etc.) -- needs the heavy n=180 G-A/G-B/tau render
  campaign under Transkun; the SUBSTRATE CAVEAT in claim-verifier-signed-d-conventions.md flags it.
