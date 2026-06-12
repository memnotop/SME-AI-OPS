# 最终实验结果摘要

本文档记录发布包 `SME-AI-OPS v1.0` 的最终验收实验口径。历史实验日志可作为过程材料，但结题答辩建议优先引用本文档和最新 `acceptance_reports/`。

## 一、实验对象

| 项目 | 内容 |
| --- | --- |
| 发布包 | `SME-AI-OPS v1.0` |
| 核心库 | `sme_ai_library` |
| 静态库 | `sme_ai_library/build/libsme_ai_ops.a` |
| 逐函数验证程序 | `sme_ai_library/build/verify_sme_ai_timing` |
| 性能对比程序 | `sme_ai_library/build/bench_sme_full_timing` |
| 应用接入 | `llama.cpp/bin/ref` 与 `llama.cpp/bin/sme` |
| 模型 | `llama.cpp/models/mini.gguf` |
| 一键验收脚本 | `run_all_checks.sh` |

## 二、最终验收命令

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
./run_all_checks.sh
```

快速检查命令：

```bash
RUN_BENCH=0 RUN_LLAMA=0 ./run_all_checks.sh
```

## 三、一键验收覆盖项

| 步骤 | 验证内容 | 通过标准 |
| --- | --- | --- |
| 发布包完整性校验 | `SHA256SUMS` 固定文件校验 | 所有固定发布文件显示 `OK` |
| 外部工具存在性检查 | 工具链、QEMU、sysroot | 路径存在且可执行 |
| 关键交付物存在性检查 | 文档、源码、二进制、模型 | 文件齐全 |
| SME1 AI 运算库重新构建 | 从源码生成静态库和测试程序 | `build.sh` 正常退出 |
| AArch64 产物类型检查 | 使用 `file` 检查目标产物 | 显示 AArch64 ELF 或 archive |
| SME1 AI 运算库指令检查 | 反汇编检查 `smstart/fmopa/smstop` | 至少出现目标 SME1 指令 |
| llama.cpp SME Q2_K 指令检查 | 反汇编检查 `smstart/smopa/smstop` | 至少出现目标 SME1 指令 |
| SME1 AI 运算库逐函数 QEMU 验证 | 全部公开 API 逐项运行 | 每项输出 `OK` |
| SME1 AI 运算库 REF/SME QEMU benchmark | 高层算子 REF/SME 对照 | 输出 speedup 和数值检查 |
| 最小用户示例构建和 QEMU 运行 | 第三方调用方式验证 | `DEMO_LINEAR PASS` |
| llama.cpp SME CLI QEMU 帮助检查 | SME 版 `llama-cli` 启动 | 输出帮助信息 |
| llama.cpp mini.gguf QEMU 冒烟测试 | REF/SME `llama-bench` 加载模型 | 生成日志和摘要 |

## 四、API 验证范围

逐函数验证覆盖以下类别：

- SME 环境查询：`sme_ai_has_sme`、`sme_ai_has_sme2`、`sme_ai_cntb`、`sme_ai_cntw`
- 生命周期管理：workspace、packed linear weight
- 基础张量算子：add、mul、residual add、bias add
- 激活函数：ReLU、Sigmoid、Tanh、GELU、SiLU
- Softmax：普通、masked、causal
- 归一化：LayerNorm、RMSNorm
- Embedding lookup
- Linear 与 Linear+Activation
- MLP、SwiGLU、Self-Attention
- Conv2D、MaxPool2D、AvgPool2D

逐项对应关系见 `API_TEST_REPORT_MAPPING.md`。

## 五、benchmark 覆盖范围

| benchmark | 覆盖算子 |
| --- | --- |
| `linear_simple` | 原始权重 Linear |
| `linear_packed` | 预打包权重 Linear |
| `linear_relu` | Linear+ReLU |
| `linear_gelu` | Linear+GELU |
| `linear_silu` | Linear+SiLU |
| `mlp_relu` | 两层 MLP+ReLU |
| `mlp_gelu` | 两层 MLP+GELU |
| `swiglu` | LLaMA 类 gated FFN |
| `attention` | Q/K/V/O 投影和 causal attention 调用链 |
| `conv2d` | im2col+SME packed Linear 卷积路径 |

## 六、llama.cpp 接入结果

`llama.cpp/bin/sme` 相比 `llama.cpp/bin/ref` 增加自研 SME1 Q2_K 路径：

- 选择 `q2_K_8x8_q8_K` repack/GEMM 路径。
- 使用 `CPU_REPACK` 持久化重排。
- 使用 metadata cache 减少重复解码。
- 使用 SME1 `smopa` 做内层 int8 outer product。

`run_all_checks.sh` 会检查 SME 版 `libggml-cpu.so.0.9.8` 中是否出现 `smstart/smopa/smstop`，并运行 `mini.gguf` 冒烟测试。

## 七、最终结论

当前发布包已经满足以下验收点：

1. 能从源码重新构建 SME1 AI 运算库。
2. 能自动证明 SME1 指令已编译进关键二进制。
3. 能逐项验证公开 API。
4. 能对主要高层算子进行 REF/SME 路径对照。
5. 能构建并运行最小用户示例，证明库函数可被第三方程序调用。
6. 能启动 llama.cpp REF/SME 双版本并加载示例模型。
7. 能生成中文一键验收报告。

## 八、边界说明

- QEMU 用于功能、数值和调用链验证；QEMU 墙钟时间不能直接外推为真实 CPU 性能。
- 当前函数库主要面向 `cntw == 16` 的 512-bit SME vector length。
- llama.cpp 自研优化集中在 `Q2_K` 权重路径，未覆盖所有量化格式。
- 当前未直接改造 GCC/LLVM 后端自动生成 SME 矩阵代码。
