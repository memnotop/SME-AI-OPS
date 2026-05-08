#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
QEMU_AARCH64=${QEMU_AARCH64:-/home/liumingjian/y_software/qemu-master/build/qemu-aarch64}
SYSROOT=${SYSROOT:-/home/liumingjian/y_software/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/aarch64-none-linux-gnu/libc}
MODEL=${MODEL:-$ROOT/models/mini.gguf}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
THREADS=${THREADS:-2}
PROMPT_TOKENS=${PROMPT_TOKENS:-1}
GEN_TOKENS=${GEN_TOKENS:-0}
REPEAT=${REPEAT:-1}

mkdir -p "$LOG_DIR"

run_bench() {
  local name=$1
  local bin=$2
  local out="$LOG_DIR/${name}.md"
  local timing="$LOG_DIR/${name}.time"

  echo "Running $name ..."
  /usr/bin/time -p env QEMU_LD_PREFIX="$SYSROOT" \
    "$QEMU_AARCH64" -cpu max,sme512=on,sme-default-vector-length=64 "$bin" \
    -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -r "$REPEAT" -t "$THREADS" -ngl 0 --no-warmup -o md \
    > "$out" 2> "$timing"
}

extract_tps() {
  local md_file=$1
  awk -F'|' '/^\| llama / {
    gsub(/^ +| +$/, "", $8);
    print $8;
    exit
  }' "$md_file"
}

extract_real() {
  local time_file=$1
  awk '/^real / {print $2 " s"; exit}' "$time_file"
}

run_bench "ref_bench_mini" "$ROOT/bin/ref/llama-bench"
run_bench "sme_bench_mini" "$ROOT/bin/sme/llama-bench"

ref_tps=$(extract_tps "$LOG_DIR/ref_bench_mini.md")
sme_tps=$(extract_tps "$LOG_DIR/sme_bench_mini.md")
ref_real=$(extract_real "$LOG_DIR/ref_bench_mini.time")
sme_real=$(extract_real "$LOG_DIR/sme_bench_mini.time")

cat > "$LOG_DIR/qemu_bench_summary.txt" <<EOF
llama.cpp mini.gguf QEMU smoke test

REF t/s: $ref_tps
SME t/s: $sme_tps
REF real: $ref_real
SME real: $sme_real

Note: QEMU is for startup/correctness smoke tests only. Final performance must be measured on the SME1 target CPU.
EOF

cat "$LOG_DIR/qemu_bench_summary.txt"
echo "Logs: $LOG_DIR"
