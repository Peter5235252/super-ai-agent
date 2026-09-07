#!/usr/bin/env bash
# Local quality gate: fmt + clippy + tests.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo clippy -- -D warnings"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> cargo test --workspace"
cargo test --workspace

echo "All checks passed."
echo "Optional (install once): cargo audit && cargo deny check"