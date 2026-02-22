#!/usr/bin/env bash
#
# Build the crescendai-shared Rust crate as an XCFramework for iOS.
#
# Usage:
#   ./build-rust-xcframework.sh
#
# Prerequisites:
#   - Rust toolchain with targets: aarch64-apple-ios, aarch64-apple-ios-sim
#   - uniffi-bindgen CLI: cargo install uniffi-bindgen-cli
#
# Output:
#   - CrescendAI/RustBridge/CrescendaiShared.xcframework
#   - CrescendAI/RustBridge/crescendai_shared.swift

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IOS_DIR="$(dirname "$SCRIPT_DIR")"
APPS_DIR="$(dirname "$IOS_DIR")"
SHARED_DIR="$APPS_DIR/shared"
BRIDGE_DIR="$IOS_DIR/CrescendAI/RustBridge"
TARGET_DIR="$APPS_DIR/target"

echo "==> Building for iOS device (aarch64-apple-ios)..."
cargo build --release --target aarch64-apple-ios --manifest-path "$SHARED_DIR/Cargo.toml" --features uniffi

echo "==> Building for iOS simulator (aarch64-apple-ios-sim)..."
cargo build --release --target aarch64-apple-ios-sim --manifest-path "$SHARED_DIR/Cargo.toml" --features uniffi

echo "==> Generating Swift bindings..."
mkdir -p "$BRIDGE_DIR"
uniffi-bindgen generate \
    "$SHARED_DIR/src/crescendai_shared.udl" \
    --language swift \
    --out-dir "$BRIDGE_DIR"

echo "==> Creating XCFramework..."
# Remove existing framework if present
rm -rf "$BRIDGE_DIR/CrescendaiShared.xcframework"

# Create headers directory for the module map
HEADERS_DIR="$BRIDGE_DIR/headers"
mkdir -p "$HEADERS_DIR"

# Move the generated header
mv "$BRIDGE_DIR/crescendai_sharedFFI.h" "$HEADERS_DIR/"

# Create module map
cat > "$HEADERS_DIR/module.modulemap" << 'MODULEMAP'
framework module crescendai_sharedFFI {
    umbrella header "crescendai_sharedFFI.h"
    export *
    module * { export * }
}
MODULEMAP

xcodebuild -create-xcframework \
    -library "$TARGET_DIR/aarch64-apple-ios/release/libcrescendai_shared.a" \
    -headers "$HEADERS_DIR" \
    -library "$TARGET_DIR/aarch64-apple-ios-sim/release/libcrescendai_shared.a" \
    -headers "$HEADERS_DIR" \
    -output "$BRIDGE_DIR/CrescendaiShared.xcframework"

# Clean up temporary headers
rm -rf "$HEADERS_DIR"

echo "==> Done! XCFramework at: $BRIDGE_DIR/CrescendaiShared.xcframework"
echo "    Swift bindings at: $BRIDGE_DIR/crescendai_shared.swift"
