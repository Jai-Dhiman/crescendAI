#!/bin/bash
set -e

echo "Building CrescendAI SSR Worker..."

# Ensure output directories exist
mkdir -p dist/client/pkg dist/worker

# 1. Install dependencies and build Tailwind CSS
echo "Installing dependencies..."
bun install

echo "Compiling Tailwind CSS..."
bun run tailwind

# 2. Build client WASM (for hydration)
echo "Building client WASM..."
cargo build --lib --target wasm32-unknown-unknown --release --features hydrate

# Install wasm-bindgen-cli if needed
if ! command -v wasm-bindgen &> /dev/null; then
    echo "Installing wasm-bindgen-cli..."
    cargo install -q wasm-bindgen-cli
fi

wasm-bindgen \
    --target web \
    --out-dir dist/client/pkg \
    --out-name crescendai \
    ./target/wasm32-unknown-unknown/release/crescendai.wasm

# Optimize WASM size (if wasm-opt is available)
if command -v wasm-opt &> /dev/null; then
    echo "Optimizing WASM..."
    wasm-opt -Oz dist/client/pkg/crescendai_bg.wasm -o dist/client/pkg/crescendai_bg.wasm
fi

# 3. Build server WASM (for Workers)
echo "Building server WASM..."
cargo install -q worker-build
worker-build --release --features ssr

# 4. Copy public assets if they exist
if [ -d "public" ]; then
    echo "Copying public assets..."
    cp -r public/* dist/client/ 2>/dev/null || true
fi

echo "Build complete!"
