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
