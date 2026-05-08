# SME-AI-OPS v1.0 发布说明

发布日期：2026-05-08

## 一、发布目标

本版本用于大学生创新训练项目终检验收，目标是形成一个可复制、可运行、可说明的软件交付包。

发布包围绕两条主线组织：

1. 面向 AArch64/SME1 的 AI 常用库函数封装。
2. 将自研 SME1 量化矩阵乘路径接入 llama.cpp 的真实模型推理链路。

## 二、主要功能

### 1. SME1 AI 运算库

本版本提供 `libsme_ai_ops.a` 静态库和公开头文件 `sme_ai_ops.h`。

主要能力：

- SME 环境查询和 512-bit vector length 检查。
- workspace 和 packed linear 权重生命周期管理。
- FP16 输入、FP32 累加/输出的 Linear、MLP、SwiGLU、Self-Attention、Conv2D。
- Softmax、Masked Softmax、Causal Softmax、LayerNorm、RMSNorm。
- ReLU、Sigmoid、Tanh、GELU、SiLU 等常用激活函数。
- MaxPool2D、AvgPool2D 和 Embedding lookup。
- REF 路径与 SME 路径对照验证。

### 2. SME1 GEMM 微内核

本版本包含 `sme1_gemm_tile_16x16.S`。

核心特点：

- 使用 SME1，不使用 SME2。
- 使用 `smstart` / `smstop` 管理 SME streaming mode。
- 使用 `fmopa` 组织 16x16 FP32 tile 外积累加。
- 当前目标为 `cntw == 16`，即 512-bit SME vector length。

### 3. llama.cpp Q2_K 接入

本版本提供 REF/SME 两套 AArch64 llama.cpp 可执行环境。

SME 版本特点：

- 面向 `Q2_K` 权重量化矩阵乘路径。
- 使用 `CPU_REPACK` 做持久化重排。
- 使用 metadata cache 减少 Q2_K scale/min 重复解码。
- 使用直接 `Q2_K x Q8_K` SME kernel，避免早期 `Q2_K -> float -> fp16` 桥接路径。
- 内层 dot 使用 SME1 `smopa`。

## 三、测试情况

发布前已在当前工作环境中完成以下验证：

- `sme_ai_library/build.sh` 可重新构建静态库和两个 AArch64 测试程序。
- `sme_ai_library/run_qemu_verify_timing.sh` 可逐项验证公开函数，输出状态为 `OK`。
- `sme_ai_library/run_qemu_full_bench.sh` 可输出 REF/SME 对比表。
- `llama.cpp/scripts/run_qemu_mini_gguf.sh` 可启动 REF/SME 两套 `llama-bench` 并生成日志。

## 四、已知限制

- QEMU 只用于功能、数值和调用链验证，不能直接代表真实目标 CPU 性能。
- `sme_ai_library` 的 SME 微内核当前要求 `cntw == 16`。
- llama.cpp 自研优化集中在 `Q2_K` 路径，未覆盖 `Q3_K`、`Q6_K` 等更多量化格式。
- Attention 中 QK^T、P*V 等内部矩阵乘仍有进一步 SME 化空间。
- 本版本未直接修改 GCC/LLVM 后端，使其自动生成新的 SME 矩阵代码路径。

## 五、后续建议

- 在真实支持 SME1 的目标 CPU 上补充 `perf`/PMU 性能数据。
- 扩展更多 ggml 量化格式，如 `Q3_K`、`Q6_K`。
- 将 self-attention 内部更多矩阵乘子步骤接入 SME 微内核。
- 将成熟微内核整理为更稳定的算子选择机制或编译器辅助优化流程。
