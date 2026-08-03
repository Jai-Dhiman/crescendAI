# One-shot corpus, catalog, and render recipes (score-library build path).
#
# These built the piece-ID catalog and exercise corpus (#17, #26, #49, #96, #97 --
# all closed). The artifacts exist; these are here so a rebuild does not have to
# re-derive the flags. Invoke as `just corpus::<recipe>`.
#
# Recipes still needed for daily dev, the score follower (#108/#133), and MIREX
# stay in the root Justfile: seed-*, fingerprint, catalog-verify, score-midi-audit,
# rebar-scores, catalog-acquire-asap, build-exercise-assets, build-scorehost.

# #49 exercise-corpus expansion: acquire the public-domain KernScores MIDIs
# (clone craigsapp humdrum repos + verovio kern->MIDI). Writes gitignored raw
# MIDI into model/data/scores/exercise_primitives/raw/. Run the #17 Mutopia
# `acquire.sh` too for the original 6 sources before a full embed rebuild.
corpus-acquire-kernscores:
    bash model/src/exercise_corpus/acquire_kernscores.sh

# #49 exercise-BOOK expansion: acquire the Chopin Op.10/25 etudes (clone
# pl-wnifc/humdrum-chopin-first-editions + verovio kern->MIDI; 24 drills).
corpus-acquire-chopin-etudes:
    bash model/src/exercise_corpus/acquire_chopin_etudes.sh

# #49 exercise-BOOK expansion: acquire Clementi's Op.42 key-preludes (18 short
# drills) from Mutopia prebuilt MIDI. No LilyPond needed.
corpus-acquire-clementi-preludes:
    bash model/src/exercise_corpus/acquire_clementi_preludes.sh

# #49: full Aria embed -> catalog -> k-NN source-purity validation over the
# exercise-book corpus. Requires the raw MIDI on disk (the corpus-acquire-*
# recipes + the #17 acquire.sh) and the Aria weights at
# model/data/weights/aria-medium-embedding. Builds model/data/exercise_primitives.db
# (gitignored) and writes results/ artifacts. SLOW (CPU Aria, ~minutes).
corpus-embed-validate:
    cd model && uv run python -m exercise_corpus.run \
        --sources src/exercise_corpus/sources.toml --output-dir data

# #49: segment ALL exercise sources (#17 + KernScores) and write the
# embed-ready manifest (model/data/embed_ready_manifest.json), STOPPING before
# the Aria embed. The GPU embed -> catalog -> validate pass is scheduled
# separately (run on the staged manifest). Requires the raw MIDI on disk
# (`just corpus::corpus-acquire-kernscores` + the #17 acquire.sh first).
corpus-segment-manifest:
    cd model && uv run python -m exercise_corpus.run \
        --sources src/exercise_corpus/sources.toml --output-dir data --segment-only

# Build local renderable .mxl assets for all catalog pieces that have an ASAP
# xml_score.musicxml (ASAP is CC-BY-NC -- LOCAL prototype use ONLY, do NOT
# deploy to production R2 or any commercial build).
# Requires: model/data/raw/asap (clone with `git clone --depth 1 https://github.com/CPJKU/asap-dataset.git model/data/raw/asap`)
# Output: model/scores/v1/<piece_id>.mxl (gitignored, regenerable)
# Runs the Verovio render gate; broken assets are reported but NOT written.
# Idempotent: pieces whose inner XML is unchanged are skipped.
build-catalog-mxl:
    cd model && uv run python -m score_library.render_assets

# Render PD-clean MEI for the catalog's KernScores pieces (Joplin/Scarlatti/
# Chopin-mazurkas) via Verovio's native Humdrum importer (~100% yield vs ~46%
# through partitura->MusicXML). Output: model/scores/v1/<piece_id>.mei
# (gitignored, regenerable). The API serves .mei in preference to .mxl. Run
# `just corpus::corpus-acquire-kernscores` first to populate the .krn clone.
render-kern-mei:
    cd model && uv run python -m score_library.render_kern_assets

# Render recognize-only Mutopia pieces (LilyPond .ly source, no clean MEI path) to
# standalone whole-piece SVG via LilyPond -dcrop. Output: scores/v1/<pid>.svg, a
# display-only tier the score-worker serves without Verovio. LOCAL-ONLY: Mutopia
# engravings are CC-BY-SA/CC-BY (not PD) -- never seed these .svg to prod R2.
# Requires `brew install lilypond` and the mutopia clone in ~/crescendai_corpus_staging.
render-mutopia-svg:
    cd model && uv run python -m score_library.mutopia_lilypond_svg

# Render recognize-only MIDI-source catalog pieces (GiantMIDI performance MIDI,
# PDMX score MIDI -- no clean MEI path) to INTERACTIVE .mxl via MuseScore 4
# headless. Output: scores/v1/<pid>.mxl, the Verovio-loadable tier (per-bar
# highlight + clip playback), preferred over a static SVG. Engine + -M import-
# quantization config were locked by a real-scorehost bake-off (issue #97).
# `source` is giantmidi|pdmx; `jobs` parallel MuseScore workers. Resumable
# (skips pieces that already have a display asset; .skip sentinel for crashers).
# Requires MuseScore 4.app + the source MIDIs in ~/crescendai_corpus_staging.
# After rendering, run `just rebar-scores` then re-verify highlighting.
render-midi-mxl source jobs="6":
    cd model && uv run python -m score_library.midi_render --source {{ source }} --jobs {{ jobs }}

# Render a stratified sample of the renderable .mei through the REAL scorehost
# Verovio worker (headed Chrome) and screenshot each. Pass a sample JSON (rows of
# {pid,total_bars,key,ts,coll}) and an output dir. Proves in-browser display, not
# just python verovio.loadData. Requires `just build-scorehost` first.
verify-mei-display sample shots_dir:
    cd apps/web && MEI_DIR="../../model/scores/v1" MEI_SAMPLE="{{sample}}" SHOTS_DIR="{{shots_dir}}" \
        bunx playwright test --config playwright.scorehost.config.ts \
        src/scorehost/mei-display.playwright.ts --headed --timeout=0 --reporter=line

# Stage net-new KernScores canon collections (Hummel preludes, Bach Art of the
# Fugue, Scriabin solo piano) as per-piece MIDI under
# ~/crescendai_corpus_staging/kernscores_midi/<repo>/ for catalog ingest. The
# until-loop is segfault-resume: Verovio's Humdrum importer SIGSEGVs on a few
# pathological **kern files, so each crash promotes that file to a permanent
# `.segfault` skip-marker and the next iteration resumes past it. Clone the
# repos first (the module aborts loudly if a clone is missing).
catalog-stage-kernscores:
    #!/usr/bin/env bash
    set -uo pipefail
    cd model
    n=0
    until uv run python -m score_library.kernscores_stage; do
        n=$((n+1)); echo "  [restart $n] Verovio SIGSEGV -- resuming past marked file"
        [ "$n" -ge 60 ] && { echo "ABORT: too many segfault restarts ($n)" >&2; exit 1; }
    done

# Expand the piece-ID catalog with the on-disk KernScores sonata/prelude
# collections (Beethoven/Mozart/Haydn/Chopin-preludes): content-dedup vs the
# existing catalog (chroma->elastic-DTW, threshold 0.2885) -> self-consistency
# ingest of net-new works -> MEI render -> re-fingerprint. Run
# `just corpus::corpus-acquire-kernscores` first to populate the .krn + MIDI staging.
catalog-expand-kernscores:
    cd model && uv run python -m score_library.kernscores_expand
    just corpus::render-kern-mei
    just fingerprint
    cd model && echo "Catalog size: $(find data/scores -maxdepth 1 -name '*.json' ! -name titles.json | wc -l | tr -d ' ') score JSONs"

# Add scores to the catalog from data/manifests/manual_scores.json (URL or local
# manual_midis/ sources). Re-ingests the whole manifest through the validation
# gate (HALTS loudly if any source is bad or off-grid), rebuilds fingerprints,
# and reports catalog size. To add a piece: append an entry to manual_scores.json
# then run this. See docs/model/10-score-library-catalog.md for the full workflow.
catalog-add:
    cd model && uv run python -m score_library.cli parse-manual --manifest data/manifests/manual_scores.json
    just fingerprint
    cd model && echo "Catalog size: $(find data/scores -maxdepth 1 -name '*.json' ! -name titles.json | wc -l | tr -d ' ') score JSONs"

# Bulk-ingest KernScores MIDI collections (Joplin, Scarlatti, Chopin mazurkas)
# via the self-consistency gate, then regenerate fingerprints.
# Source MIDIs are read from ~/crescendai_corpus_staging/kernscores_midi/.
# Fails loudly if any entire collection yields zero passes.
catalog-add-kernscores:
    cd model && uv run python -m score_library.kernscores_bulk
    just fingerprint
    cd model && echo "Catalog size: $(find data/scores -maxdepth 1 -name '*.json' ! -name titles.json | wc -l | tr -d ' ') score JSONs"

# Acquire the Mutopia second-wave keyboard corpus (clone sources + polite download).
catalog-acquire-mutopia:
    bash model/src/score_library/acquire_mutopia.sh

# Cross-performance recall + open-set (leave-one-out) verification of the piece-ID
# margin gate against held-out ASAP performance MIDIs (1066 perfs / 242 works).
# Runs the FROZEN production gate (chroma top-K -> pitch-only elastic-DTW -> lock
# iff margin>=0.0935). Ground truth is gate-independent (derive_piece_id + score-MIDI
# oracle). Reports cross-performance recall, leave-one-out false-accept decomposed
# into genuine-different-piece vs duplicate-of-true, and a threshold sweep.
# Requires ASAP (run catalog-acquire-asap first); fails loud if metadata.csv missing.
catalog-pieceid-crossperf-verify:
    cd model && PYTHONUNBUFFERED=1 uv run python -m score_library.pieceid_crossperf_verify --per-work 0 --note-cap 600

# Comprehensive production-fidelity piece-ID eval across edge-case axes the clean-MIDI
# cross-performance harness cannot see: B mid-piece starts, C transposition, D same-composer
# confusion matrix, E confidence calibration, G notes-to-lock latency. Builds the 11K catalog
# + gate ONCE and runs every axis by transforming the query notes (frozen gate, threshold is a
# parameter). CIs cluster-bootstrap by work. Axes A/F are gated siblings (corpus::catalog-pieceid-amt-axis).
catalog-pieceid-comprehensive-eval:
    cd model && PYTHONUNBUFFERED=1 uv run python -m score_library.pieceid_comprehensive_eval --axes b,c,d,e,g --per-work 1 --note-cap 600 --threshold 0.13

# Axis A -- the #1 production-fidelity gap: real audio -> production AMT service -> the frozen
# piece-ID gate, PAIRED against the clean-MIDI recognition of the same performance. Maps ASAP
# perfs to MAESTRO audio (519 matched), slices 15s chunks with 15s context (production windowing),
# POSTs to the AMT service, stitches notes, runs the gate. GATED: requires MAESTRO audio rehydrated
# (--maestro-dir; offloaded ~34GB) AND the AMT service up (`just amt`). Fails loud if either missing.
catalog-pieceid-amt-axis maestro_dir limit="50":
    cd model && PYTHONUNBUFFERED=1 uv run python -m score_library.pieceid_amt_axis --maestro-dir {{maestro_dir}} --limit {{limit}} --opening-seconds 90 --threshold 0.13

# Axis A by STREAMING MAESTRO audio per-perf from HF (no local MAESTRO needed; each
# WAV is downloaded, transcribed, then deleted -> peak disk ~2-3GB for the full 519).
# Resumable (re-run continues from the incremental output). PREREQ: the AMT service
# must be up (`just amt`, or `cd apps/inference && CRESCEND_DEVICE=cpu uv run amt/amt_local_server.py`
# -- use cpu to avoid contending with MPS training). limit=0 runs all 519 (CPU ~1-1.5 days;
# pick a smaller limit for an overnight pilot).
catalog-pieceid-amt-axis-stream limit="100":
    cd model && PYTHONUNBUFFERED=1 uv run python -m score_library.pieceid_amt_axis --stream-hf ddPn08/maestro-v3.0.0 --limit {{limit}} --opening-seconds 90 --threshold 0.13

# AUTORESEARCH verify command: drives the experimental knob-board
# (score_library/pieceid_experimental.py -- the ONLY file an autoresearch loop edits)
# over held-out ASAP perfs vs the 11K catalog and prints a single scalar
# `AUTORESEARCH_METRIC: <float>` rewarding opening + mid-piece recognition while
# penalizing genuine open-set false-accepts above the 5% bar. Parsed-catalog +
# label caches make each iteration ~50s. Pass CATALOG/CACHE/LABEL paths as needed;
# defaults assume the catalog is in-worktree (data/scores) with sibling cache files.
catalog-pieceid-autoresearch catalog_cache="data/evals/piece_id/catalog_cache.pkl" label_cache="data/evals/piece_id/labels_cache.json":
    cd model && PYTHONUNBUFFERED=1 uv run python -m score_library.pieceid_autoresearch_eval --catalog-cache {{catalog_cache}} --label-cache {{label_cache}}

# Catalog-wide AMT-aware duplicate detection (NON-DESTRUCTIVE manifest, nothing deleted).
# Greedy nearest-keep: drops a piece only on a DIRECT match to an already-kept higher-
# priority piece (engraved>pdmx>giantmidi, most-notes). Source-aware threshold (engraved
# 0.2885, AMT-involved 0.40); splits high-confidence (<=0.2885) from medium review-pool.
# Writes data/evals/piece_id/dedup_manifest.json -- input for the deferred cleanliness pass.
catalog-dedup-scan:
    cd model && PYTHONUNBUFFERED=1 uv run python -m score_library.dedup_scan --note-cap 600 --top-k 8

# Add net-new Mutopia keyboard pieces to the catalog: instrument-filter ->
# content semantic dedup vs existing catalog -> self-consistency ingest ->
# re-fingerprint. Run `just corpus::catalog-acquire-mutopia` first.
catalog-add-mutopia:
    cd model && uv run python -m score_library.mutopia_filter
    cd model && uv run python -m score_library.mutopia_dedup
    cd model && uv run python -m score_library.mutopia_ingest
    just fingerprint
    cd model && echo "Catalog size: $(find data/scores -maxdepth 1 -name '*.json' ! -name titles.json | wc -l | tr -d ' ') score JSONs"

# Disaster-recovery: regenerate per-piece score JSONs from the fingerprint
# (piece_index.json = authoritative piece set) + staging source MIDIs, via
# parse_score_midi. Covers ~92% (giantmidi/pdmx/mutopia with a clean piece_id<->
# MIDI map); kern-family pieces recover from R2 or their ingest recipes. Resumable.
regen-scores fingerprint jobs="8":
    cd model && uv run python -m score_library.regen_scores --fingerprint {{ fingerprint }} --jobs {{ jobs }}
