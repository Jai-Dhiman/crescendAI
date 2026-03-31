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
    echo "  MuQ:  http://localhost:8000"
    echo "  AMT:  http://localhost:8001"
    echo "  API:  http://localhost:8787"
    echo "  Web:  http://localhost:3000"
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

# Start API worker (Rust, Cloudflare Workers, port 8787)
api:
    cd apps/api-rust && npx wrangler dev

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

# Run API cargo check (Rust)
check-api:
    cd apps/api-rust && cargo check

# Run API tests (Rust)
test-api:
    cd apps/api-rust && cargo test

# Start Hono API worker (TypeScript, port 8787)
api-new:
    cd apps/api && npx wrangler dev --local

# Run Hono API tests
test-api-new:
    cd apps/api && bun run test -- --run

# Type-check Hono API
check-api-new:
    cd apps/api && bun run typecheck

# Run web type check
check-web:
    cd apps/web && bun run typecheck

# Deploy API worker to production (Rust)
deploy-api:
    cd apps/api-rust && npx wrangler deploy

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

# Apply D1 migrations (local, Rust API)
migrate-local:
    cd apps/api-rust && npx wrangler d1 migrations apply DB --local

# Apply D1 migrations (production, Rust API)
migrate-prod:
    cd apps/api-rust && npx wrangler d1 migrations apply DB --remote

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
