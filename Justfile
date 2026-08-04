# CrescendAI Development Commands
#
# One-shot corpus/catalog/render recipes live in corpus.just: `just corpus::<recipe>`
# Usage: just <recipe>
# List all recipes: just --list

mod corpus 'corpus.just'

# Default recipe: start all dev services
default:
    @just --list

# Start all dev services (MuQ + AMT + API + Web)
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT

    echo "Starting CrescendAI dev environment..."
    echo ""

    just muq &
    just amt &
    just api &
    just web &
    wait

# Start MuQ inference server (quality scoring, port 8000)
# Note: uv run (without python) triggers PEP 723 inline dependency resolution
# Checkpoint override: real ablation/optimized_weights heads are absent locally;
# point at the r32 sweep folds so all 4 heads load.
muq:
    cd apps/inference && uv run muq/muq_local_server.py --checkpoint-dir ../../model/data/checkpoints/a1_max_sweep/A1max_r32_L7-12_ls0.1

# Start Transkun inference server (transcription, port 8001)
amt:
    cd apps/inference && uv run amt/amt_local_server.py --port 8001

# Start API worker (Hono, Cloudflare Workers, port 8787)
api:
    cd apps/api && bun run dev

# Start web dev server (TanStack Start, port 3000)
web:
    cd apps/web && bun run dev

# Start without inference (web + API only, uses production HF endpoints)
dev-light:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT

    echo "Starting CrescendAI (web + API only, no local inference)..."
    echo "  API:  http://localhost:8787"
    echo "  Web:  http://localhost:3000"
    echo ""

    just api &
    just web &
    wait

# Start MuQ only (no AMT transcription, faster startup for scores-only testing)
dev-muq:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT

    echo "Starting CrescendAI (MuQ + API + Web, no AMT)..."
    echo "  MuQ:  http://localhost:8000"
    echo "  API:  http://localhost:8787"
    echo "  Web:  http://localhost:3000"
    echo ""

    just muq &
    just api &
    just web &
    wait

# Seed local R2 with score files from model/scores/v1/ (run once after fresh checkout)
seed-scores:
    #!/usr/bin/env bash
    set -euo pipefail
    for f in model/scores/v1/*.mxl; do
        key="scores/v1/$(basename "$f")"
        echo "Seeding $key"
        cd apps/api && wrangler r2 object put "crescendai-bucket/$key" --file="../../$f" --local && cd ../..
    done

# Seed PD-clean Verovio MEI render assets (model/scores/v1/*.mei) into R2 at
# scores/v1/<piece_id>.mei (the format getPieceData prefers). `mode` is "local"
# (wrangler dev) or "remote" (PROD R2 -- authorized for PD-clean MEI only;
# never seed the CC-BY-NC ASAP .mxl to prod). Optional `filter` seeds only
# piece_ids with that prefix (e.g. `just seed-mei remote scriabin.`) so a
# catalog increment can push just its new assets instead of all 472.
seed-mei mode="local" filter="":
    #!/usr/bin/env bash
    set -euo pipefail
    flag="--local"; [ "{{mode}}" = "remote" ] && flag="--remote"
    count=0
    for f in model/scores/v1/*.mei; do
        base="$(basename "$f")"
        if [ -n "{{filter}}" ] && [[ "$base" != "{{filter}}"* ]]; then continue; fi
        cd apps/api && wrangler r2 object put "crescendai-bucket/scores/v1/$base" \
            --file="../../$f" --content-type "application/mei+xml" $flag >/dev/null && cd ../..
        count=$((count+1))
    done
    echo "Seeded $count .mei into $flag R2 (filter='{{filter}}')."

# Seed the LilyPond-rendered Mutopia .svg display assets into LOCAL wrangler R2.
# LOCAL-ONLY by design: these engravings are CC-BY-SA/CC-BY (not PD) -- there is
# deliberately NO remote mode here (mirrors the CC-BY-NC ASAP .mxl local-only rule).
seed-mutopia-svg filter="mutopia.":
    #!/usr/bin/env bash
    set -euo pipefail
    count=0
    for f in model/scores/v1/*.svg; do
        base="$(basename "$f")"
        if [ -n "{{filter}}" ] && [[ "$base" != "{{filter}}"* ]]; then continue; fi
        cd apps/api && wrangler r2 object put "crescendai-bucket/scores/v1/$base" \
            --file="../../$f" --content-type "image/svg+xml" --local >/dev/null && cd ../..
        count=$((count+1))
    done
    echo "Seeded $count .svg into LOCAL R2 (filter='{{filter}}')."

# Generate the v2 piece-ID index (chroma + chord-events) from the score library
fingerprint:
    cd model && uv run python -m score_library.cli fingerprint --scores-dir data/scores --output-dir data/fingerprints

# Audit score-MIDI coverage for Aria score-conditioning (issue #73). Writes
# data/manifests/score_midi_coverage.json; exits non-zero on any uncovered T1/T5 piece.
score-midi-audit:
    uv run model/scripts/audit_score_midi_coverage.py --validate-midi

# Seed the v2 piece-ID artifact into LOCAL wrangler R2 for `wrangler dev`.
# Run `just fingerprint` first to produce model/data/fingerprints/piece_index.json.
seed-fingerprint:
    cd apps/api && wrangler r2 object put "crescendai-bucket/fingerprint/v2/piece_index.json" \
        --file="../../model/data/fingerprints/piece_index.json" --local

# Seed per-piece score JSONs into LOCAL wrangler R2 at scores/v1/{pieceId}.json.
# After a piece-ID lock the SessionBrain DO reads scores/v1/{pieceId}.json (env.SCORES)
# to drive chroma-DTW score-following (alignChunkChroma). The key is the score JSON's
# `piece_id` (== the filename stem), which matches the piece_index.json piece_id the gate
# returns. Pass an optional filter to seed a single piece (e.g. `just seed-score-json beethoven.fur_elise`);
# with no arg, seeds the whole catalog. Source: model/data/scores/*.json (regen via `just fingerprint`'s
# upstream score-library recipes). Mirrors `seed-scores` (which seeds .mxl for rendering, a different key).
seed-score-json filter="":
    #!/usr/bin/env bash
    set -euo pipefail
    count=0
    for f in model/data/scores/*.json; do
        base="$(basename "$f")"
        [ "$base" = "titles.json" ] && continue
        pid="${base%.json}"
        if [ -n "{{filter}}" ] && [ "$pid" != "{{filter}}" ]; then continue; fi
        cd apps/api && wrangler r2 object put "crescendai-bucket/scores/v1/$base" --file="../../$f" --local >/dev/null && cd ../..
        count=$((count+1))
        echo "Seeded scores/v1/$base"
    done
    echo "Seeded $count score JSON(s) into local R2."

# Regenerate the committed renderable .mxl assets from the committed primitive
# .xml (model/data/scores/exercise_primitives/*.xml -> model/data/exercise_primitives/mxl/*.mxl).
# Deterministic + idempotent; raises naming any .xml that fails partitura load.
build-exercise-assets:
    cd model && uv run python -c "from pathlib import Path; from exercise_corpus.build_render_assets import build; mp = Path('..') / 'apps' / 'api' / 'src' / 'services' / 'exercise_primitives_manifest.json'; print(f'built {len(build(manifest_path=mp))} assets + manifest at {mp}')"

# Re-bar every score JSON that has a paired engraved .mei OR .mxl so its bar grid
# matches the displayed measures (total_bars == Verovio measure count, PPQ derived
# per-piece from the JSON: kern=120, GiantMIDI=384, PDMX=480). Notes are preserved
# exactly, so the piece-ID fingerprint is unchanged. Fixes live score-highlighting
# alignment for repeat-heavy / mis-segmented pieces (~210/618 had a drifted grid).
# Re-run `just fingerprint` is NOT required (note set unchanged) but harmless.
rebar-scores backup_dir="":
    cd model && uv run python -m score_library.rebar_from_mei \
        --scores-dir data/scores --mei-dir scores/v1 \
        {{ if backup_dir != "" { "--backup-dir " + backup_dir } else { "" } }}

# Seed the committed exercise-primitive .mxl assets into LOCAL wrangler R2 at
# scores/v1/{primitive_id}.mxl so the UNCHANGED GET /api/scores/:pieceId/data
# endpoint serves them for corpus_drill rendering. Flat keyspace: primitive ids
# (hanon_001) cannot collide with real ASAP piece slugs. Run `just build-exercise-assets`
# first. Mirrors `seed-scores` (real-piece .mxl) and `seed-fingerprint`.
seed-exercise-assets:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    count=0
    # Drills carry tiered assets: .mei (Verovio-native, Chopin etudes) preferred by
    # getPieceData, .mxl (wrapped MusicXML) for the rest. Seed both with the right
    # content-type so scores/v1/{id}.{mei,mxl} serves identically to the piece library.
    for f in model/data/exercise_primitives/assets/*.mei model/data/exercise_primitives/assets/*.mxl; do
        base="$(basename "$f")"
        case "$base" in
            *.mei) ct="application/mei+xml" ;;
            *.mxl) ct="application/vnd.recordare.musicxml+zip" ;;
            *) echo "ERROR: unexpected asset extension: $base" >&2; exit 1 ;;
        esac
        ( cd apps/api && wrangler r2 object put "crescendai-bucket/scores/v1/$base" \
            --file="../../$f" --content-type "$ct" --local >/dev/null )
        count=$((count+1))
        echo "Seeded scores/v1/$base ($ct)"
    done
    if [ "$count" -eq 0 ]; then
        echo "ERROR: no assets found in model/data/exercise_primitives/assets/ — run 'just build-exercise-assets' first" >&2
        exit 1
    fi
    echo "Seeded $count exercise-primitive asset(s) into local R2."

# Verify all 16 labeled eval pieces have a non-trivial, monotonic catalog entry
catalog-verify:
    cd model && uv run python -c "from score_library.catalog_coverage import check_coverage, CANONICAL_MAP; from src.paths import Scores; import sys; f=check_coverage(Scores.root, CANONICAL_MAP); print(chr(10).join(f) if f else 'PASS'); sys.exit(1 if f else 0)"

# Acquire ASAP (held-out performances) for cross-performance piece-ID verification.
catalog-acquire-asap:
    cd model && test -d data/raw/asap || git clone --depth 1 https://github.com/CPJKU/asap-dataset.git data/raw/asap

# Run model tests
test-model:
    cd model && uv run python -m pytest tests/ -v

# Run full A1-Max sweep with Chunk A diagnostics (GPU recommended: ~10h on A100)
train-sweep:
    cd model && uv run python -m model_improvement.a1_max_sweep

# --- HF Jobs cloud training (issue #74) ---

# Local overnight baseline diagnostic run (best config only, 4 prefetch workers, ~8h on MPS)
train-sweep-local:
    cd model && uv run python -m model_improvement.a1_max_sweep --top-n-configs 1 --num-workers 4

# Run API type check
check-api:
    cd apps/api && bun run typecheck

# Run API tests
test-api:
    cd apps/api && bun run test -- --run && bunx vitest run --config vitest.node.config.ts

# Run the piece-identify Rust unit + parity tests (native target).
test-piece-id:
    cd apps/api/src/wasm/piece-identify && cargo test

# Run web type check
check-web:
    cd apps/web && bun run typecheck

# Deploy API worker to production
deploy-api:
    cd apps/api && bun run deploy

# --- AMT Container ---

# Start AMT container dev server (Cloudflare Containers)
amt-container-dev:
    cd apps/inference/amt && bun run dev

# Deploy AMT container to production
amt-container-deploy:
    cd apps/inference/amt && bun run deploy

# Build AMT container Docker image
amt-container-build:
    cd apps/inference/amt && docker build \
      --build-arg CHECKPOINT_PATH=./checkpoint.safetensors \
      -t crescendai-amt .

# Check AMT container health via SSH into running container
amt-container-health:
    cd apps/inference/amt && wrangler containers ssh amt-0 -- curl -s http://localhost:8080/health

# Generate Drizzle migration SQL
migrate-generate:
    cd apps/api && bun run generate

# Apply Drizzle migrations to production Postgres
migrate-prod:
    cd apps/api && bun run migrate

# --- E2E UI Session Test (issue #68) ---

# Drive one real recording through the live local stack (MuQ+AMT -> glm V6 synthesis)
# and verify the synthesis renders correctly in the web UI via Playwright.
# Requires: just dev (or just dev-muq) + just seed-fingerprint already running.
# Default recording: model/data/evals/practice_eval/nocturne_op9no2/audio/_aySCutsVVQ.wav
e2e-ui-session recording=("model/data/evals/practice_eval/nocturne_op9no2/audio/_aySCutsVVQ.wav") piece="nocturne_op9no2":
    cd apps/evals && uv run python -m e2e_ui_session \
        --recording "../../{{recording}}" \
        --piece-slug "{{piece}}" \
        --screenshot /tmp/e2e-ui-session.png

# --- E2E Pipeline Eval ---

# Run full E2E pipeline eval (cache -> pipeline -> analyze)
eval-e2e: eval-cache eval-pipeline eval-analyze

# Generate missing inference cache for T5 corpus (requires just muq + just amt running)
eval-cache:
    cd apps/evals && uv run python -m inference.eval_runner --auto-t5

# Run pipeline eval on T5 corpus (requires just api running)
eval-pipeline:
    cd apps/evals && uv run python -m pipeline.practice_eval.eval_practice --scenarios t5

# Analyze eval results
eval-analyze:
    cd apps/evals && uv run python -m pipeline.practice_eval.analyze_e2e --report reports/practice_eval.json

# Generate T5 scenario files from manifests
eval-scenarios:
    cd apps/evals && uv run python -m pipeline.practice_eval.generate_t5_scenarios

# Test playbook YAML shape (apps/shared/teacher-style/)
test-playbook-shape:
    cd apps/shared/teacher-style && uv run --no-project --with pyyaml --with pytest pytest test_playbook_shape.py -v

# Compile apps/shared/teacher-style/playbook.yaml -> apps/api/src/lib/playbook.json
compile-playbook:
    uv run --with pyyaml python apps/shared/scripts/compile_playbook.py

# Check that all Swift files under apps/ios/ are registered in project.pbxproj
check-ios:
    #!/usr/bin/env bash
    set -euo pipefail
    pbxproj="apps/ios/CrescendAI.xcodeproj/project.pbxproj"
    missing=()
    while IFS= read -r f; do
        name=$(basename "$f")
        if ! grep -qF "/* $name */" "$pbxproj"; then
            missing+=("$f")
        fi
    done < <(find apps/ios/CrescendAI apps/ios/CrescendAITests -name "*.swift")
    if [ ${#missing[@]} -eq 0 ]; then
        echo "OK: all Swift files are registered in project.pbxproj"
    else
        echo "ERROR: the following Swift files are on disk but not in project.pbxproj:"
        for f in "${missing[@]}"; do echo "  $f"; done
        exit 1
    fi

# CI sync check: fail if compiled JSON is stale
check-playbook-sync:
    uv run --with pyyaml python apps/shared/scripts/compile_playbook.py --check

# Lint API (biome)
lint-api:
    cd apps/api && bun run lint

# Lint web (biome)
lint-web:
    cd apps/web && bun run lint

# Deterministic checks for the mechanical rules in docs/standards/rules.json
check-standards:
    python3 scripts/standards_check.py --all

# Point git at the version-controlled hooks in .githooks/ (run once per clone)
install-hooks:
    git config core.hooksPath .githooks
    @echo "installed: pre-commit (staged lint + standards), pre-push (typecheck + tests)"

# Build dtw_chunk_cli release binary so chroma-eval-verify hits its 120s budget on warm cache.
# Run once after a clean checkout; idempotent thereafter.
chroma-eval-prebuild:
    cd apps/api/src/wasm/score-analysis && cargo build --release --bin dtw_chunk_cli

# CI/dev sanity check: smoke path (--skip-dtw). Exercises sampler + pseudo-truth
# + aggregator + sidecar without needing real audio or the Rust DTW binary.
# Primary scalar (100.0) is by construction -- not a real measurement.
# Use this in pre-commit hooks and fast CI gates.
chroma-eval-verify-smoke:
    cd model && uv run python -m chroma_dtw_eval.verify \
        --baseline data/evals/chroma_dtw_fixtures/smoke_baseline.json \
        --manifest data/evals/chroma_dtw_fixtures/manifest.json \
        --skip-dtw

# Full chroma-DTW eval using real audio + Rust DTW binary. This is the real
# measurement loop -- run before /autoresearch dispatch or ratcheting baseline.
# Requires:
#   model/data/evals/practice_eval/bach_prelude_c_wtc1/audio/*.wav
#   model/data/evals/pseudo_truth/bach_prelude_c_wtc1/
# If data is missing, run `just amt-regen-pseudo-truth` first;
# see docs/implementation/2026-06-01-amt-pseudo-truth-pilot.md
# 120s budget assumes Rust DTW binary is pre-built; run `just chroma-eval-prebuild` once.
chroma-eval-verify:
    #!/usr/bin/env bash
    set -euo pipefail
    AUDIO_DIR="model/data/evals/practice_eval/bach_prelude_c_wtc1/audio"
    PT_DIR="model/data/evals/pseudo_truth/bach_prelude_c_wtc1"
    if [ ! -d "$AUDIO_DIR" ] || [ -z "$(ls -A "$AUDIO_DIR" 2>/dev/null)" ]; then
        echo "ERROR: real audio not found at $AUDIO_DIR" >&2
        echo "Run \`just amt-regen-pseudo-truth <piece> <video_id>\` first;" >&2
        echo "see docs/implementation/2026-06-01-amt-pseudo-truth-pilot.md" >&2
        exit 1
    fi
    if [ ! -d "$PT_DIR" ] || [ -z "$(ls -A "$PT_DIR" 2>/dev/null)" ]; then
        echo "ERROR: pseudo-truth not found at $PT_DIR" >&2
        echo "Run \`just amt-regen-pseudo-truth <piece> <video_id>\` first;" >&2
        echo "see docs/implementation/2026-06-01-amt-pseudo-truth-pilot.md" >&2
        exit 1
    fi
    cd model && uv run python -m chroma_dtw_eval.verify \
        --baseline data/evals/chroma_dtw/baseline.json \
        --manifest data/evals/chroma_dtw_fixtures/manifest.json

# Promote the latest sidecar to baseline. Refuses to write on regression.
chroma-eval-ratchet:
    cd model && uv run python -m chroma_dtw_eval.ratchet \
        --from data/evals/chroma_dtw/last_run.json \
        --to data/evals/chroma_dtw/baseline.json

# Regenerate AMT pseudo-truth cache for a single (piece, video_id).
# Usage: just amt-regen-pseudo-truth <piece_id> <video_id>
amt-regen-pseudo-truth piece video_id:
    cd model && uv run python -m chroma_dtw_eval.amt_regen \
        --piece {{piece}} --video-id {{video_id}}

# Chroma-identification feasibility harness (Issue #21).
# Downloads real audio on cache miss; use --holdout to configure open-set probe.
# Run `just piece-id-feasibility-acquire` first to populate audio cache.
piece-id-feasibility:
    cd model && uv run python -m piece_id_eval.cli

# Acquire audio for all 16 practice_eval pieces (yt-dlp required).
piece-id-feasibility-acquire slug video_id:
    cd model && uv run python -c "from pathlib import Path; from piece_id_eval.acquire import acquire_audio; out = acquire_audio('{{video_id}}', Path('data/evals/practice_eval/{{slug}}/audio')); print('Downloaded:', out)"

# Build the standalone scorehost web bundle (required before iOS build)
build-scorehost:
    cd apps/web && bun run build:scorehost

# Exercise-routing eval: drive all practice_eval WAVs through real local inference.
# Requires `just dev` (MuQ:8000 + AMT:8001 + API:8787) and `just seed-fingerprint`.
exercise-routing-eval:
    cd apps/evals && uv run python -m pipeline.exercise_routing.eval_routing

# Exercise-routing eval smoke: validate wiring without live services (CI-safe).
exercise-routing-eval-smoke:
    cd apps/evals && uv run python -m pipeline.exercise_routing.eval_routing --skip-inference

# Promote last_run.json -> baseline.json after a deliberate routing improvement.
exercise-routing-ratchet:
    cd apps/evals && uv run python -m pipeline.exercise_routing.ratchet_baseline

# Full visible-browser E2E eval: seed canary facts, drive a real session,
# open headed Chrome, assert synthesis + memory recall via scripted chat.
# Requires: just dev (API:8787, Web:3000, MuQ:8000), just seed-fingerprint.
# Use --reply-timeout 180000 on a cold Workers AI stack (glm cold-start >90s).
e2e-full-session *args:
    cd apps/evals && uv run python e2e_full_session.py --reply-timeout 180000 {{args}}
