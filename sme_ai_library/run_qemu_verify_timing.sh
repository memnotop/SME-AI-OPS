#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
QEMU=${QEMU:-/home/liumingjian/y_software/qemu-master/build/qemu-aarch64}
CPU=${CPU:-max,sme512=on,sme-default-vector-length=64}
ITER_SCALE=${ITER_SCALE:-1}
CASES=${CASES:-"
sme_ai_has_sme
sme_ai_has_sme2
sme_ai_cntb
sme_ai_cntw
sme_ai_workspace_init
sme_ai_workspace_destroy
sme_ai_packed_linear_weight_init
sme_ai_packed_linear_weight_destroy
sme_ai_add_f32
sme_ai_mul_f32
sme_ai_residual_add_f32
sme_ai_add_bias_rowwise_f32
sme_ai_relu_inplace_f32
sme_ai_sigmoid_inplace_f32
sme_ai_tanh_inplace_f32
sme_ai_gelu_inplace_f32
sme_ai_silu_inplace_f32
sme_ai_softmax_rowwise_f32
sme_ai_softmax_masked_rowwise_f32
sme_ai_softmax_causal_rowwise_f32
sme_ai_layernorm_f32
sme_ai_rmsnorm_f32
sme_ai_embedding_lookup_f16f32
sme_ai_max_pool2d_nchw_f32
sme_ai_avg_pool2d_nchw_f32
sme_ai_linear_ref_f16f32
sme_ai_linear_sme_simple_f16f32
sme_ai_linear_sme_packed_f16f32
sme_ai_linear_relu_ref_f16f32
sme_ai_linear_relu_sme_packed_f16f32
sme_ai_linear_gelu_ref_f16f32
sme_ai_linear_gelu_sme_packed_f16f32
sme_ai_linear_silu_ref_f16f32
sme_ai_linear_silu_sme_packed_f16f32
sme_ai_mlp_relu_ref_f16f32
sme_ai_mlp_relu_sme_packed_f16f32
sme_ai_mlp_gelu_ref_f16f32
sme_ai_mlp_gelu_sme_packed_f16f32
sme_ai_swiglu_ref_f16f32
sme_ai_swiglu_sme_packed_f16f32
sme_ai_self_attention_ref_f16f32
sme_ai_self_attention_sme_packed_f16f32
sme_ai_conv2d_ref_nchw_f16f32
sme_ai_conv2d_sme_im2col_nchw_f16f32
"}

if [[ ! -x "$QEMU" ]]; then
  echo "QEMU not found or not executable: $QEMU" >&2
  echo "Set QEMU=/path/to/qemu-aarch64 and run again." >&2
  exit 1
fi

if [[ ! -x "$ROOT/build/verify_sme_ai_timing" ]]; then
  echo "Missing executable: $ROOT/build/verify_sme_ai_timing" >&2
  echo "Run: bash build.sh" >&2
  exit 1
fi

run_one() {
  local case_name=$1
  local start_ns end_ns output status iters elems checksum

  start_ns=$(date +%s%N)
  if ! output=$("$QEMU" -cpu "$CPU" "$ROOT/build/verify_sme_ai_timing" --case "$case_name" --iter-scale "$ITER_SCALE" --no-header 2>&1); then
    echo "$output" >&2
    return 1
  fi
  end_ns=$(date +%s%N)

  status=$(awk '{print $2}' <<<"$output")
  iters=$(awk '{print $3}' <<<"$output")
  elems=$(awk '{print $4}' <<<"$output")
  checksum=$(awk '{print $6}' <<<"$output")
  if [[ "$status" != "OK" ]]; then
    echo "$output" >&2
    return 1
  fi

  elapsed_ms=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.3f", (e - s) / 1000000.0 }')
  printf "%-42s %-4s %7s %8s %11s % .6e\n" "$case_name" "$status" "$iters" "$elems" "$elapsed_ms" "$checksum"
}

echo "QEMU_VERIFY_TIME"
echo "CPU $CPU"
echo "ITER_SCALE $ITER_SCALE"
printf "%-42s %-4s %7s %8s %11s %13s\n" "CASE" "OK" "ITERS" "ELEMS" "QEMU_ms" "CHECKSUM"

for case_name in $CASES; do
  run_one "$case_name"
done
