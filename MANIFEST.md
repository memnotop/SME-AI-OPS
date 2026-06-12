# 交付物清单

## 一、发布包根目录

| 路径 | 说明 |
| --- | --- |
| `VERSION` | 发布版本信息 |
| `README.md` | 发布包总说明 |
| `QUICKSTART.md` | 快速开始与常用命令 |
| `RELEASE_NOTES.md` | v1.0 发布说明 |
| `CHANGELOG.md` | 版本变更记录 |
| `CONTRIBUTING.md` | 后续开发说明 |
| `API_TEST_REPORT_MAPPING.md` | 公开 API、测试用例和验收报告对应表 |
| `GOAL_TRACEABILITY.md` | 前期/中期目标与当前成果的对应表 |
| `FINAL_EXPERIMENT_SUMMARY.md` | 最终实验结果摘要 |
| `LICENSE.md` | 许可与第三方组件说明 |
| `MANIFEST.md` | 本交付清单 |
| `SHA256SUMS` | 固定发布文件 SHA256 校验值，不包含可再生成的 `build/`、`logs/` 和 `acceptance_reports/` |
| `run_all_checks.sh` | 一键验收脚本，生成中文验收报告 |

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

API、测试用例和验收报告的逐项对应关系见：

```text
API_TEST_REPORT_MAPPING.md
```

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

## 五、扩展说明文档

| 路径 | 说明 |
| --- | --- |
| `docs/architecture.md` | 软件架构说明 |
| `docs/build_and_run.md` | 构建与运行说明 |
| `docs/api_reference.md` | API 参考摘要 |
| `docs/test_methodology.md` | 测试方法说明 |
| `docs/performance_summary.md` | 性能结果说明 |
| `docs/limitations_and_future.md` | 局限与后续工作 |

## 六、最小用户示例

| 路径 | 说明 |
| --- | --- |
| `examples/README.md` | 示例说明 |
| `examples/demo_linear.c` | 最小 Linear 调用示例 |
| `examples/build_demo_linear.sh` | 示例构建脚本 |
| `examples/run_qemu_demo_linear.sh` | 示例 QEMU 运行脚本 |

## 七、验收时建议展示的命令

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

也可以在发布包根目录直接运行完整验收：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
./run_all_checks.sh
```
