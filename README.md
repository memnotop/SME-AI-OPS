# SME-AI-OPS v1.0 软件发布包

本目录是“面向人工智能算法中矩阵加速的飞腾处理器软件适配与优化”项目的正式软件交付包。

本发布包的核心交付物是：

- `sme_ai_library/`：SME1 AI 运算库，包含头文件、源码、静态库、benchmark 和逐函数验证程序。
- `llama.cpp/`：llama.cpp 的 REF/SME 两套 AArch64 可执行环境，包含 `llama-bench`、`llama-cli`、动态库、脚本和 `mini.gguf` 示例模型。
- `doc/`：SME1 指令原理说明。
- `MANIFEST.md`：交付物清单。
- `QUICKSTART.md`：快速开始。
- `RELEASE_NOTES.md`：发布说明。
- `LICENSE.md`：许可与第三方组件说明。
- `SHA256SUMS`：文件完整性校验值。

## 适用范围

本软件用于验证和展示 ARM AArch64/SME1 环境下 AI 常用矩阵与张量算子的库函数封装，以及自研 SME1 量化矩阵乘路径接入 llama.cpp 的工程流程。

本发布包主要面向结题验收、后续真机测试和继续调优使用。QEMU 可用于功能和调用链验证，但 QEMU 墙钟时间不能直接作为真实飞腾/目标 CPU 的性能结论。

## 快速运行

一键验收：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
./run_all_checks.sh
```

SME AI 运算库：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
bash build.sh
./run_qemu_verify_timing.sh
./run_qemu_full_bench.sh
```

llama.cpp 冒烟测试：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/llama.cpp
./scripts/run_qemu_cli_help.sh
./scripts/run_qemu_mini_gguf.sh
```

目标 CPU 上运行时，可直接使用：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/llama.cpp
./scripts/run_target_bench_mini.sh
VERSION=sme PROMPT="Hello." N_PREDICT=16 ./scripts/run_target_cli_mini.sh
```

## 发布边界

- 本发布包自研优化只使用 SME1，不依赖 SME2。
- `sme_ai_library` 当前微内核要求 `cntw == 16`，即 512-bit SME vector length。
- `llama.cpp` 的自研优化集中在 `Q2_K` 权重量化矩阵乘路径，不覆盖模型中的全部量化格式。
- 本发布包不宣称已经完成 GCC/LLVM 编译器后端改造；编译器方向以工具链验证、构建支持和原理说明为主。

## 建议验收顺序

1. 阅读 `MANIFEST.md` 确认交付物。
2. 运行 `./run_all_checks.sh` 生成中文验收报告。
3. 查看 `sme_ai_library/include/sme_ai_ops.h` 了解公开 API。
4. 查看 `llama.cpp/doc/SME_Q2K_OPTIMIZATION.md` 了解 llama.cpp 接入方式。
5. 查看 `SHA256SUMS` 做完整性校验。
