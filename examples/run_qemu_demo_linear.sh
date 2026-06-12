#!/usr/bin/env bash
set -euo pipefail

# 通过 QEMU 运行最小 Linear 示例。

ROOT=$(cd "$(dirname "$0")/.." && pwd)
QEMU=${QEMU:-/home/liumingjian/y_software/qemu-master/build/qemu-aarch64}
CPU=${CPU:-max,sme512=on,sme-default-vector-length=64}

if [[ ! -x "$QEMU" ]]; then
  echo "QEMU not found or not executable: $QEMU" >&2
  exit 1
fi

if [[ ! -x "$ROOT/examples/build/demo_linear" ]]; then
  echo "Missing example binary, run examples/build_demo_linear.sh first." >&2
  exit 1
fi

"$QEMU" -cpu "$CPU" "$ROOT/examples/build/demo_linear"
