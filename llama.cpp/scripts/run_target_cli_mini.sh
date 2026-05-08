#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERSION=${VERSION:-sme}
MODEL=${MODEL:-$ROOT/models/mini.gguf}
THREADS=${THREADS:-2}
N_PREDICT=${N_PREDICT:-16}
PROMPT=${PROMPT:-"Hello, introduce yourself briefly."}

case "$VERSION" in
  ref|sme) ;;
  *)
    echo "VERSION must be ref or sme" >&2
    exit 2
    ;;
esac

echo "Running llama-cli ($VERSION) ..."
"$ROOT/bin/$VERSION/llama-cli" -m "$MODEL" -p "$PROMPT" -n "$N_PREDICT" -t "$THREADS" -ngl 0 --no-display-prompt
