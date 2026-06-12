#!/usr/bin/env bash
set -euo pipefail

# 构建最小 Linear 示例。

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT:-/home/liumingjian/y_software/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu}
CC=${CC:-$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-gcc}

if [[ ! -x "$CC" ]]; then
  echo "AArch64 GCC not found or not executable: $CC" >&2
  exit 1
fi

if [[ ! -f "$ROOT/sme_ai_library/build/libsme_ai_ops.a" ]]; then
  echo "Missing libsme_ai_ops.a, building sme_ai_library first ..."
  (cd "$ROOT/sme_ai_library" && TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" bash build.sh)
fi

mkdir -p "$ROOT/examples/build"

"$CC" \
  -O2 \
  -static \
  -std=gnu11 \
  -Wall \
  -Wextra \
  -I"$ROOT/sme_ai_library/include" \
  -march=armv9-a+sve+sme \
  "$ROOT/examples/demo_linear.c" \
  "$ROOT/sme_ai_library/build/libsme_ai_ops.a" \
  -lm \
  -o "$ROOT/examples/build/demo_linear"

file "$ROOT/examples/build/demo_linear"
