#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT:-/home/liumingjian/y_software/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu}
CC=${CC:-$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-gcc}
AR=${AR:-$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-ar}

CFLAGS=(
  -O3
  -static
  -std=gnu11
  -Wall
  -Wextra
  -I"$ROOT/include"
  -march=armv9-a+sve+sme
)

rm -rf "$ROOT/build"
mkdir -p "$ROOT/build/obj"

"$CC" "${CFLAGS[@]}" -c "$ROOT/src/sme_ai_ops.c" -o "$ROOT/build/obj/sme_ai_ops.o"
"$CC" "${CFLAGS[@]}" -c "$ROOT/src/sme1_gemm_tile_16x16.S" -o "$ROOT/build/obj/sme1_gemm_tile_16x16.o"
"$AR" rcs "$ROOT/build/libsme_ai_ops.a" "$ROOT/build/obj/sme_ai_ops.o" "$ROOT/build/obj/sme1_gemm_tile_16x16.o"
"$CC" "${CFLAGS[@]}" "$ROOT/test_bench/bench_sme_full_timing.c" "$ROOT/build/libsme_ai_ops.a" -lm -o "$ROOT/build/bench_sme_full_timing"
"$CC" "${CFLAGS[@]}" "$ROOT/test_bench/verify_sme_ai_timing.c" "$ROOT/build/libsme_ai_ops.a" -lm -o "$ROOT/build/verify_sme_ai_timing"

rm -rf "$ROOT/build/obj"

file "$ROOT/build/bench_sme_full_timing"
file "$ROOT/build/verify_sme_ai_timing"
