#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Applying D1 migrations..."
for migration in migrations/0*.sql; do
    echo "  $migration"
    bunx wrangler d1 execute crescendai-db --local --file="$migration" 2>/dev/null || true
done

echo "Starting wrangler dev..."
bunx wrangler dev --local --port 8787 &
WRANGLER_PID=$!

cleanup() {
    echo "Stopping wrangler dev (PID $WRANGLER_PID)..."
    kill $WRANGLER_PID 2>/dev/null || true
    wait $WRANGLER_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8787/health > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Server failed to start within 60 seconds."
        exit 1
    fi
    sleep 1
done

echo "Running tests..."
bun test tests/exercises.test.ts
EXIT_CODE=$?

exit $EXIT_CODE
