#!/usr/bin/env bash
set -Eeuo pipefail

# SME-AI-OPS v1.0 一键验收脚本。
# 作用：自动完成构建、文件类型检查、SME 指令反汇编检查、QEMU 功能验证、
#      QEMU benchmark 和 llama.cpp 冒烟测试，并生成中文 Markdown 验收报告。

ROOT=$(cd "$(dirname "$0")" && pwd)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_BASE=${REPORT_BASE:-"$ROOT/acceptance_reports"}
REPORT_DIR=${REPORT_DIR:-"$REPORT_BASE/$TIMESTAMP"}
LOG_DIR="$REPORT_DIR/logs"
REPORT="$REPORT_DIR/acceptance_report.md"
RESULTS="$REPORT_DIR/results.tsv"

TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT:-/home/liumingjian/y_software/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu}
QEMU=${QEMU:-/home/liumingjian/y_software/qemu-master/build/qemu-aarch64}
QEMU_AARCH64=${QEMU_AARCH64:-"$QEMU"}
SYSROOT=${SYSROOT:-"$TOOLCHAIN_ROOT/aarch64-none-linux-gnu/libc"}
OBJDUMP=${OBJDUMP:-"$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-objdump"}
GCC=${GCC:-"$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-gcc"}
AR=${AR:-"$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-ar"}

# 默认执行完整验收。现场时间紧时可以用 RUN_LLAMA=0 或 RUN_BENCH=0 跳过耗时步骤。
RUN_SHA256=${RUN_SHA256:-1}
RUN_BENCH=${RUN_BENCH:-1}
RUN_LLAMA=${RUN_LLAMA:-1}
ITER_SCALE=${ITER_SCALE:-1}

mkdir -p "$LOG_DIR"
: > "$RESULTS"

TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

safe_name() {
  printf "%s" "$1" | tr ' /:' '___' | tr -cd '[:alnum:]_.-'
}

append_report_header() {
  cat > "$REPORT" <<EOF
# SME-AI-OPS v1.0 一键验收报告

生成时间：$(date +"%Y-%m-%d %H:%M:%S %z")

发布包目录：

\`\`\`text
$ROOT
\`\`\`

## 验收环境

| 项目 | 值 |
| --- | --- |
| 工具链根目录 | \`$TOOLCHAIN_ROOT\` |
| AArch64 GCC | \`$GCC\` |
| AArch64 AR | \`$AR\` |
| AArch64 objdump | \`$OBJDUMP\` |
| QEMU | \`$QEMU\` |
| QEMU sysroot | \`$SYSROOT\` |
| RUN_SHA256 | \`$RUN_SHA256\` |
| RUN_BENCH | \`$RUN_BENCH\` |
| RUN_LLAMA | \`$RUN_LLAMA\` |
| ITER_SCALE | \`$ITER_SCALE\` |

## 验收步骤

EOF
}

record_result() {
  local status=$1
  local name=$2
  local log=$3
  local note=$4

  printf "%s\t%s\t%s\t%s\n" "$status" "$name" "$log" "$note" >> "$RESULTS"
  cat >> "$REPORT" <<EOF
### $name

- 状态：$status
- 日志：\`$log\`
- 说明：$note

EOF
}

run_step() {
  local name=$1
  local workdir=$2
  local command=$3
  local note=${4:-"见日志。"}
  local log="$LOG_DIR/$(printf "%02d" "$((TOTAL + 1))")_$(safe_name "$name").log"

  TOTAL=$((TOTAL + 1))
  printf "[%02d] %s ... " "$TOTAL" "$name"
  if (cd "$workdir" && bash -lc "$command") > "$log" 2>&1; then
    PASSED=$((PASSED + 1))
    printf "通过\n"
    record_result "通过" "$name" "$log" "$note"
    return 0
  fi

  FAILED=$((FAILED + 1))
  printf "失败\n"
  record_result "失败" "$name" "$log" "$note"
  return 1
}

skip_step() {
  local name=$1
  local note=$2
  local log="-"

  TOTAL=$((TOTAL + 1))
  SKIPPED=$((SKIPPED + 1))
  printf "[%02d] %s ... 跳过\n" "$TOTAL" "$name"
  record_result "跳过" "$name" "$log" "$note"
}

append_summary() {
  cat >> "$REPORT" <<EOF
## 汇总

| 指标 | 数量 |
| --- | ---: |
| 总步骤 | $TOTAL |
| 通过 | $PASSED |
| 失败 | $FAILED |
| 跳过 | $SKIPPED |

EOF

  if [[ "$FAILED" -eq 0 ]]; then
    cat >> "$REPORT" <<EOF
结论：本次一键验收通过。发布包可以完成源码构建、SME 指令存在性检查、QEMU 功能验证和配置范围内的运行测试。

EOF
  else
    cat >> "$REPORT" <<EOF
结论：本次一键验收未通过。请优先查看状态为“失败”的步骤日志。

EOF
  fi

  cat >> "$REPORT" <<EOF
## 重要说明

- QEMU 测试用于确认程序可运行、数值检查可通过、SME 路径可被调用。
- QEMU 墙钟时间受仿真器影响较大，不能直接作为真实飞腾/目标 CPU 的最终性能结论。
- 若现场只需快速检查，可使用 \`RUN_LLAMA=0 RUN_BENCH=0 ./run_all_checks.sh\` 缩短时间。

EOF
}

append_report_header

if [[ "$RUN_SHA256" == "1" ]]; then
  run_step "发布包完整性校验" "$ROOT" "sha256sum -c SHA256SUMS" "构建前校验发布包内静态文件是否与 SHA256SUMS 一致。" || true
else
  skip_step "发布包完整性校验" "RUN_SHA256=$RUN_SHA256，按配置跳过。"
fi

run_step "外部工具存在性检查" "$ROOT" \
  "test -x '$GCC' && test -x '$AR' && test -x '$OBJDUMP' && test -x '$QEMU' && test -d '$SYSROOT' && '$GCC' --version | head -n 1 && '$QEMU' --version | head -n 1" \
  "检查交叉工具链、QEMU 和 sysroot 是否存在。" || true

run_step "关键交付物存在性检查" "$ROOT" \
  "test -f VERSION && test -f README.md && test -f MANIFEST.md && test -f QUICKSTART.md && test -f RELEASE_NOTES.md && test -f LICENSE.md && test -f sme_ai_library/include/sme_ai_ops.h && test -f sme_ai_library/src/sme_ai_ops.c && test -f sme_ai_library/src/sme1_gemm_tile_16x16.S && test -f llama.cpp/bin/ref/llama-bench && test -f llama.cpp/bin/sme/llama-bench && test -f llama.cpp/models/mini.gguf" \
  "检查发布说明、函数库源码、llama.cpp 双版本二进制和示例模型是否齐全。" || true

if run_step "SME1 AI 运算库重新构建" "$ROOT/sme_ai_library" \
  "TOOLCHAIN_ROOT='$TOOLCHAIN_ROOT' CC='$GCC' AR='$AR' bash build.sh" \
  "重新生成 libsme_ai_ops.a、bench_sme_full_timing 和 verify_sme_ai_timing。"; then
  BUILD_OK=1
else
  BUILD_OK=0
fi

if [[ "$BUILD_OK" == "1" ]]; then
  run_step "AArch64 产物类型检查" "$ROOT" \
    "file sme_ai_library/build/libsme_ai_ops.a sme_ai_library/build/bench_sme_full_timing sme_ai_library/build/verify_sme_ai_timing llama.cpp/bin/sme/libggml-cpu.so.0.9.8" \
    "确认构建产物和 llama.cpp SME 核心库为 AArch64 目标产物。" || true

  run_step "SME1 AI 运算库指令检查" "$ROOT" \
    "'$OBJDUMP' -d sme_ai_library/build/bench_sme_full_timing | tee '$REPORT_DIR/sme_ai_library_sme_instructions.txt' | grep -E 'smstart|fmopa|smstop' && grep -q 'fmopa' '$REPORT_DIR/sme_ai_library_sme_instructions.txt'" \
    "反汇编确认函数库测试二进制中包含 smstart/fmopa/smstop。" || true

  run_step "llama.cpp SME Q2_K 指令检查" "$ROOT" \
    "'$OBJDUMP' -d llama.cpp/bin/sme/libggml-cpu.so.0.9.8 | tee '$REPORT_DIR/llama_cpp_sme_instructions.txt' | grep -E 'smstart|smopa|smstop' && grep -q 'smopa' '$REPORT_DIR/llama_cpp_sme_instructions.txt'" \
    "反汇编确认 llama.cpp SME 版核心库中包含 smstart/smopa/smstop。" || true

  run_step "SME1 AI 运算库逐函数 QEMU 验证" "$ROOT/sme_ai_library" \
    "QEMU='$QEMU' ITER_SCALE='$ITER_SCALE' ./run_qemu_verify_timing.sh" \
    "逐项调用公开 API，所有函数应输出 OK。" || true

  if [[ "$RUN_BENCH" == "1" ]]; then
    run_step "SME1 AI 运算库 REF/SME QEMU benchmark" "$ROOT/sme_ai_library" \
      "QEMU='$QEMU' ITER_SCALE='$ITER_SCALE' ./run_qemu_full_bench.sh" \
      "输出 REF_ms、SME_ms 和 SPEEDUP，用于仿真环境下的路径对比。" || true
  else
    skip_step "SME1 AI 运算库 REF/SME QEMU benchmark" "RUN_BENCH=$RUN_BENCH，按配置跳过。"
  fi
else
  skip_step "AArch64 产物类型检查" "构建失败，跳过依赖构建产物的检查。"
  skip_step "SME1 AI 运算库指令检查" "构建失败，跳过依赖构建产物的检查。"
  skip_step "SME1 AI 运算库逐函数 QEMU 验证" "构建失败，跳过依赖构建产物的检查。"
  skip_step "SME1 AI 运算库 REF/SME QEMU benchmark" "构建失败，跳过依赖构建产物的检查。"
fi

if [[ "$RUN_LLAMA" == "1" ]]; then
  run_step "llama.cpp SME CLI QEMU 帮助检查" "$ROOT/llama.cpp" \
    "QEMU_AARCH64='$QEMU_AARCH64' SYSROOT='$SYSROOT' VERSION=sme ./scripts/run_qemu_cli_help.sh" \
    "确认 SME 版 llama-cli 可以在 QEMU 下启动并输出帮助信息。" || true

  run_step "llama.cpp mini.gguf QEMU 冒烟测试" "$ROOT/llama.cpp" \
    "QEMU_AARCH64='$QEMU_AARCH64' SYSROOT='$SYSROOT' LOG_DIR='$REPORT_DIR/llama_logs' THREADS=2 PROMPT_TOKENS=1 GEN_TOKENS=0 REPEAT=1 ./scripts/run_qemu_mini_gguf.sh" \
    "确认 REF/SME 两套 llama-bench 均能加载 mini.gguf 并输出日志。" || true
else
  skip_step "llama.cpp SME CLI QEMU 帮助检查" "RUN_LLAMA=$RUN_LLAMA，按配置跳过。"
  skip_step "llama.cpp mini.gguf QEMU 冒烟测试" "RUN_LLAMA=$RUN_LLAMA，按配置跳过。"
fi

append_summary

printf "\n验收报告：%s\n" "$REPORT"
printf "步骤结果：%s\n" "$RESULTS"
printf "日志目录：%s\n" "$LOG_DIR"

if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
