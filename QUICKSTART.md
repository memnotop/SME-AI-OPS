# 快速开始

本文档给出发布包的最短复现流程。所有命令均假设发布包路径为：

```text
/home/liumingjian/dachuang/SME-AI-OPS-v1.0
```

## 一、环境要求

当前脚本默认使用以下工具路径：

```text
/home/liumingjian/y_software/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu
/home/liumingjian/y_software/qemu-master/build/qemu-aarch64
```

如果工具不在默认路径，可通过环境变量覆盖：

```bash
export TOOLCHAIN_ROOT=/path/to/arm-gnu-toolchain
export QEMU=/path/to/qemu-aarch64
export QEMU_AARCH64=/path/to/qemu-aarch64
```

## 二、构建 SME1 AI 运算库

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
bash build.sh
```

构建成功后应生成：

```text
build/libsme_ai_ops.a
build/bench_sme_full_timing
build/verify_sme_ai_timing
```

这些产物均为 AArch64 目标文件或可执行文件。

## 三、运行逐函数验证

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
./run_qemu_verify_timing.sh
```

预期输出包含：

```text
QEMU_VERIFY_TIME
CPU max,sme512=on,sme-default-vector-length=64
```

每一行函数状态应为 `OK`。

## 四、运行 REF/SME 对比 benchmark

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
./run_qemu_full_bench.sh
```

输出字段含义：

- `CASE`：测试算子。
- `REF_ms`：参考实现耗时。
- `SME_ms`：SME 路径耗时。
- `SPEEDUP`：`REF_ms / SME_ms`，大于 `1.0x` 表示 SME 路径在当前测试中更快。

注意：QEMU 数据用于验证功能和运行路径，不作为真实硬件最终性能结论。

## 五、运行 llama.cpp 冒烟测试

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/llama.cpp
./scripts/run_qemu_cli_help.sh
./scripts/run_qemu_mini_gguf.sh
```

`run_qemu_mini_gguf.sh` 会在 `llama.cpp/logs/` 下生成：

```text
ref_bench_mini.md
ref_bench_mini.time
sme_bench_mini.md
sme_bench_mini.time
qemu_bench_summary.txt
```

## 六、目标 CPU 上运行

在支持 SME1 的 AArch64 目标 CPU 上，可直接运行：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
./build/verify_sme_ai_timing
./build/bench_sme_full_timing
```

llama.cpp：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/llama.cpp
./scripts/run_target_bench_mini.sh
VERSION=sme PROMPT="Hello." N_PREDICT=16 ./scripts/run_target_cli_mini.sh
```

## 七、完整性校验

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
sha256sum -c SHA256SUMS
```

如果固定发布文件未被修改，应显示 `OK`。`SHA256SUMS` 不校验可再生成的 `sme_ai_library/build/`、`llama.cpp/logs/` 和 `acceptance_reports/`，这些内容分别由构建步骤、运行脚本和验收报告重新生成。

## 八、一键验收

发布包根目录提供一键验收脚本：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
./run_all_checks.sh
```

脚本会自动执行：

- 发布包完整性校验；
- 外部工具存在性检查；
- 关键交付物存在性检查；
- SME1 AI 运算库重新构建；
- AArch64 产物类型检查；
- SME 指令反汇编检查；
- SME1 AI 运算库逐函数 QEMU 验证；
- SME1 AI 运算库 REF/SME QEMU benchmark；
- llama.cpp CLI 和 `mini.gguf` 冒烟测试。

脚本会在 `acceptance_reports/` 下生成中文 Markdown 验收报告和完整日志。

公开 API 与测试项的对应关系见：

```text
API_TEST_REPORT_MAPPING.md
```

如果现场只需要快速检查，可跳过 benchmark 和 llama.cpp 冒烟测试：

```bash
RUN_BENCH=0 RUN_LLAMA=0 ./run_all_checks.sh
```
