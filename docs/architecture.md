# 软件架构说明

## 一、总体架构

`SME-AI-OPS v1.0` 按四层组织：

| 层次 | 内容 | 目录 |
| --- | --- | --- |
| 环境层 | AArch64 GNU 工具链、QEMU SME 运行环境 | `run_all_checks.sh`、`QUICKSTART.md` |
| 微内核层 | SME1 16x16 GEMM tile、`fmopa/smopa` 指令使用 | `sme_ai_library/src/sme1_gemm_tile_16x16.S`、`doc/SME1_INSTRUCTION_PRINCIPLES.md` |
| 库函数层 | AI 常用算子 API、workspace、packed weight | `sme_ai_library/include/sme_ai_ops.h`、`sme_ai_library/src/sme_ai_ops.c` |
| 应用层 | llama.cpp REF/SME 双版本，Q2_K 路径接入 | `llama.cpp/` |

## 二、SME AI 运算库

`sme_ai_library` 是本发布包的软件主体。它包含：

- 公开头文件：`include/sme_ai_ops.h`
- C 实现：`src/sme_ai_ops.c`
- SME1 汇编微内核：`src/sme1_gemm_tile_16x16.S`
- 构建脚本：`build.sh`
- 逐函数验证：`verify_sme_ai_timing`
- 性能对比：`bench_sme_full_timing`

库函数设计上分为三类：

1. 参考实现：函数名中带 `ref`，用于正确性对照。
2. SME simple 实现：调用时临时整理权重，适合调试。
3. SME packed 实现：预先打包权重，适合重复推理。

## 三、workspace 与 packed weight

`sme_ai_workspace` 用于统一管理临时缓冲区，避免每次算子调用都重复 `malloc/free`。

`sme_ai_packed_linear_weight` 用于把 Linear 权重预处理成 SME 微内核更容易读取的布局。对同一层权重重复推理时，应优先使用 packed 版本。

## 四、llama.cpp 接入

`llama.cpp/bin/ref` 是参考版本，`llama.cpp/bin/sme` 是包含自研 SME1 Q2_K 路径的版本。

SME 版本的主要变化：

- 为 `Q2_K` tensor 选择 `q2_K_8x8_q8_K` 路径。
- 使用 `CPU_REPACK` 做权重持久化重排。
- 为 Q2_K metadata 建立 cache。
- 内层 dot 使用 SME1 `smopa`。

详细说明见 `llama.cpp/doc/SME_Q2K_OPTIMIZATION.md`。
