# 交付物清单

## 一、发布包根目录

| 路径 | 说明 |
| --- | --- |
| `VERSION` | 发布版本信息 |
| `README.md` | 发布包总说明 |
| `QUICKSTART.md` | 快速开始与常用命令 |
| `RELEASE_NOTES.md` | v1.0 发布说明 |
| `LICENSE.md` | 许可与第三方组件说明 |
| `MANIFEST.md` | 本交付清单 |
| `SHA256SUMS` | 文件 SHA256 校验值 |

## 二、SME1 AI 运算库

| 路径 | 说明 |
| --- | --- |
| `sme_ai_library/README.md` | SME1 AI 运算库说明 |
| `sme_ai_library/build.sh` | AArch64 静态库与测试程序构建脚本 |
| `sme_ai_library/include/sme_ai_ops.h` | 对外公开 API 头文件 |
| `sme_ai_library/src/sme_ai_ops.c` | AI 算子库主体实现 |
| `sme_ai_library/src/sme1_gemm_tile_16x16.S` | SME1 16x16 GEMM tile 汇编微内核 |
| `sme_ai_library/test_bench/bench_sme_full_timing.c` | REF/SME 性能对比 benchmark |
| `sme_ai_library/test_bench/verify_sme_ai_timing.c` | 公开函数逐项验证程序 |
| `sme_ai_library/run_qemu_full_bench.sh` | QEMU 下运行性能对比脚本 |
| `sme_ai_library/run_qemu_verify_timing.sh` | QEMU 下运行逐函数验证脚本 |
| `sme_ai_library/build/libsme_ai_ops.a` | 已构建的 AArch64 静态库 |
| `sme_ai_library/build/bench_sme_full_timing` | 已构建的 benchmark 二进制 |
| `sme_ai_library/build/verify_sme_ai_timing` | 已构建的逐函数验证二进制 |

公开 API 覆盖范围包括：

- SME 环境查询：`sme_ai_has_sme`、`sme_ai_has_sme2`、`sme_ai_cntb`、`sme_ai_cntw`
- 生命周期管理：workspace 初始化/释放、packed linear 权重初始化/释放
- 基础张量算子：加法、乘法、残差加、逐行 bias
- 激活函数：ReLU、Sigmoid、Tanh、GELU、SiLU
- 归一化和 softmax：Softmax、Masked Softmax、Causal Softmax、LayerNorm、RMSNorm
- Transformer/LLM 常用结构：Embedding、Linear、MLP、SwiGLU、Self-Attention
- CNN 常用结构：Conv2D、MaxPool2D、AvgPool2D

## 三、llama.cpp REF/SME 双版本环境

| 路径 | 说明 |
| --- | --- |
| `llama.cpp/README.md` | llama.cpp REF/SME 交付说明 |
| `llama.cpp/bin/ref/` | 参考版本 AArch64 可执行文件和动态库 |
| `llama.cpp/bin/sme/` | 含自研 SME1 Q2_K 路径的 AArch64 可执行文件和动态库 |
| `llama.cpp/models/mini.gguf` | TinyLlama 1.1B Chat Q2_K Medium 示例模型 |
| `llama.cpp/scripts/run_qemu_cli_help.sh` | QEMU 下检查 `llama-cli --help` |
| `llama.cpp/scripts/run_qemu_mini_gguf.sh` | QEMU 下运行 mini.gguf 冒烟测试 |
| `llama.cpp/scripts/run_target_bench_mini.sh` | 目标 CPU 上运行 REF/SME benchmark |
| `llama.cpp/scripts/run_target_cli_mini.sh` | 目标 CPU 上运行 REF/SME CLI 推理 |
| `llama.cpp/logs/` | 已有运行日志和摘要 |
| `llama.cpp/doc/SME_Q2K_OPTIMIZATION.md` | Q2_K SME1 优化说明 |

## 四、原理文档

| 路径 | 说明 |
| --- | --- |
| `doc/SME1_INSTRUCTION_PRINCIPLES.md` | SME1 核心指令与本项目使用方式说明 |

## 五、验收时建议展示的命令

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
bash build.sh
./run_qemu_verify_timing.sh
./run_qemu_full_bench.sh
```

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/llama.cpp
./scripts/run_qemu_cli_help.sh
./scripts/run_qemu_mini_gguf.sh
```
