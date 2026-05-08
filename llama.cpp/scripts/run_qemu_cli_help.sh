#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
QEMU_AARCH64=${QEMU_AARCH64:-/home/liumingjian/y_software/qemu-master/build/qemu-aarch64}
SYSROOT=${SYSROOT:-/home/liumingjian/y_software/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/aarch64-none-linux-gnu/libc}
VERSION=${VERSION:-sme}

case "$VERSION" in
  ref|sme) ;;
  *)
    echo "VERSION must be ref or sme" >&2
    exit 2
    ;;
esac

env QEMU_LD_PREFIX="$SYSROOT" \
  "$QEMU_AARCH64" -cpu max,sme512=on,sme-default-vector-length=64 \
  "$ROOT/bin/$VERSION/llama-cli" --help
