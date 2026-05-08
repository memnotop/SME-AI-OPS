#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-$ROOT/models/mini.gguf}
THREADS=${THREADS:-2}
PROMPT_TOKENS=${PROMPT_TOKENS:-1}
GEN_TOKENS=${GEN_TOKENS:-0}
REPEAT=${REPEAT:-1}

echo "Running REF llama-bench ..."
"$ROOT/bin/ref/llama-bench" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -r "$REPEAT" -t "$THREADS" -ngl 0 --no-warmup -o md

echo "Running SME llama-bench ..."
"$ROOT/bin/sme/llama-bench" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -r "$REPEAT" -t "$THREADS" -ngl 0 --no-warmup -o md
