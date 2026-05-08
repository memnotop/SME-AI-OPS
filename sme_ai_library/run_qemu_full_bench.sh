#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
QEMU=${QEMU:-/home/liumingjian/y_software/qemu-master/build/qemu-aarch64}
CPU=${CPU:-max,sme512=on,sme-default-vector-length=64}
ITER_SCALE=${ITER_SCALE:-1}
CASES=${CASES:-"linear_simple linear_packed linear_relu linear_gelu linear_silu mlp_relu mlp_gelu swiglu attention conv2d"}

if [[ ! -x "$QEMU" ]]; then
  echo "QEMU not found or not executable: $QEMU" >&2
  echo "Set QEMU=/path/to/qemu-aarch64 and run again." >&2
  exit 1
fi

if [[ ! -x "$ROOT/build/bench_sme_full_timing" ]]; then
  echo "Missing executable: $ROOT/build/bench_sme_full_timing" >&2
  echo "Run: bash build.sh" >&2
  exit 1
fi

run_one() {
  local case_name=$1
  local mode=$2
  local start_ns end_ns output

  start_ns=$(date +%s%N)
  if ! output=$("$QEMU" -cpu "$CPU" "$ROOT/build/bench_sme_full_timing" --case "$case_name" --mode "$mode" --iter-scale "$ITER_SCALE" 2>&1); then
    echo "$output" >&2
    return 1
  fi
  end_ns=$(date +%s%N)

  if ! grep -q " pass " <<<"$output"; then
    echo "$output" >&2
    return 1
  fi

  awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.3f", (e - s) / 1000000.0 }'
}

echo "QEMU_FULL_TIME_COMPARE"
echo "CPU $CPU"
echo "ITER_SCALE $ITER_SCALE"
printf "%-18s %12s %12s %10s\n" "CASE" "REF_ms" "SME_ms" "SPEEDUP"

for case_name in $CASES; do
  ref_ms=$(run_one "$case_name" ref)
  sme_ms=$(run_one "$case_name" sme)
  speedup=$(awk -v r="$ref_ms" -v s="$sme_ms" 'BEGIN { if (s > 0) printf "%.2fx", r / s; else printf "0.00x" }')
  printf "%-18s %12s %12s %10s\n" "$case_name" "$ref_ms" "$sme_ms" "$speedup"
done
