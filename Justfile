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
muq:
    cd apps/inference && uv run python muq_local_server.py

# Start Aria-AMT inference server (transcription, port 8001)
amt:
    cd apps/inference && uv run python amt_local_server.py --port 8001

# Start API worker (Cloudflare Workers, port 8787)
api:
    cd apps/api && npx wrangler dev

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

# Run API cargo check
check-api:
    cd apps/api && cargo check

# Run API tests
test-api:
    cd apps/api && cargo test

# Run web type check
check-web:
    cd apps/web && bun run typecheck

# Deploy API worker to production
deploy-api:
    cd apps/api && npx wrangler deploy

# Apply D1 migrations (local)
migrate-local:
    cd apps/api && npx wrangler d1 migrations apply DB --local

# Apply D1 migrations (production)
migrate-prod:
    cd apps/api && npx wrangler d1 migrations apply DB --remote
