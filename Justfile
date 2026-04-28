# CrescendAI Development Commands
# Usage: just <recipe>
# List all recipes: just --list

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
muq:
    cd apps/inference && uv run muq/muq_local_server.py

# Start Aria-AMT inference server (transcription, port 8001)
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

# Generate fingerprint index + rerank features from score library
fingerprint:
    cd model && uv run python -m score_library.cli fingerprint --scores-dir data/scores --output-dir data/fingerprints

# Run model tests
test-model:
    cd model && uv run python -m pytest tests/ -v

# Run full A1-Max sweep with Chunk A diagnostics (GPU recommended: ~10h on A100)
train-sweep:
    cd model && uv run python -m model_improvement.a1_max_sweep

# Local overnight baseline diagnostic run (best config only, 4 prefetch workers, ~8h on MPS)
train-sweep-local:
    cd model && uv run python -m model_improvement.a1_max_sweep --top-n-configs 1 --num-workers 4

# Stamp A1-Max baseline diagnostics into all Wave 1 plan files (run after either sweep)
stamp-baseline:
    cd model && uv run python scripts/stamp_baseline_diagnostics.py

# Run API type check
check-api:
    cd apps/api && bun run typecheck

# Run API tests
test-api:
    cd apps/api && bun run test -- --run && bunx vitest run --config vitest.node.config.ts

# Run web type check
check-web:
    cd apps/web && bun run typecheck

# Deploy API worker to production
deploy-api:
    cd apps/api && wrangler deploy

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
    cd apps/shared/teacher-style && uv run --with pyyaml --with pytest pytest test_playbook_shape.py -v

# Compile apps/shared/teacher-style/playbook.yaml -> apps/api/src/lib/playbook.json
compile-playbook:
    uv run --with pyyaml python apps/shared/scripts/compile_playbook.py

# CI sync check: fail if compiled JSON is stale
check-playbook-sync:
    uv run --with pyyaml python apps/shared/scripts/compile_playbook.py --check

# Lint API (biome)
lint-api:
    cd apps/api && bun run lint

# Lint web (biome)
lint-web:
    cd apps/web && bun run lint
