# 目标-成果-证据对应表

本文档用于把前期申请书、中期汇报中的目标，与当前发布包中的软件成果和验收证据逐项对应，避免结题材料只描述结果而缺少可核验依据。

## 一、总体结论

本项目原始目标包括“飞腾/ARM 架构与 SME 指令研究”“编译器与库函数开发”“搭建验证环境”“接入人工智能算法并验证性能”。当前发布包将可交付成果收敛为：

1. AArch64/SME1 工具链与 QEMU 验证环境。
2. SME1 AI 运算库 `sme_ai_library`。
3. llama.cpp REF/SME 双版本运行环境。
4. 一键验收脚本、API-测试-报告对应表和最终实验材料。

需要明确边界：当前成果不宣称已经直接修改 GCC/LLVM 后端，使其自动生成新的 SME 矩阵代码；编译器方向以交叉工具链验证、SME 编译标志、构建流程和原理说明为主。

## 二、前期目标对应表

| 前期/中期目标 | 当前交付成果 | 证据文件或命令 | 完成度 | 边界与后续工作 |
| --- | --- | --- | --- | --- |
| 学习典型 AI 算法，识别矩阵/向量运算瓶颈 | 中期完成 Llama.cpp、YOLOv8、Qwen/QuteTTS 等模型部署和 perf 热点分析；最终聚焦 Linear、MLP、SwiGLU、Attention、Conv2D | `FINAL_EXPERIMENT_SUMMARY.md`；`API_TEST_REPORT_MAPPING.md` | 已完成主线 | 可补更完整的真机 perf/PMU 火焰图 |
| 研究飞腾/ARM AArch64 架构和 SME 指令 | 提供 SME1 原理说明、`fmopa/smopa` 使用说明、反汇编检查 | `doc/SME1_INSTRUCTION_PRINCIPLES.md`；`run_all_checks.sh` 的“SME 指令检查”步骤 | 已完成软件侧验证 | 目标 CPU 微架构调度仍需真机实测 |
| 搭建适用于飞腾/ARM 的模拟验证环境 | 使用 AArch64 GNU 工具链和 QEMU SME user-mode/full-system 工作流 | `QUICKSTART.md`；`run_all_checks.sh`；`acceptance_reports/<时间戳>/acceptance_report.md` | 已完成 | QEMU 只用于功能和调用链验证，不代表真机性能 |
| 编写或优化库函数，提高 AI 算法矩阵运算效率 | 形成 `libsme_ai_ops.a`，公开 API 覆盖基础张量、激活、Softmax、Norm、Embedding、Linear、MLP、SwiGLU、Attention、Conv2D、Pooling | `sme_ai_library/include/sme_ai_ops.h`；`API_TEST_REPORT_MAPPING.md`；`run_qemu_verify_timing.sh` | 已完成可交付库函数 | 当前微内核面向 `cntw == 16`，可继续扩展更多向量长度 |
| 将核心函数集成到现有人工智能算法中验证 | 提供 llama.cpp REF/SME 双版本，SME 版接入 Q2_K x Q8_K 自研 SME1 路径 | `llama.cpp/doc/SME_Q2K_OPTIMIZATION.md`；`llama.cpp/scripts/run_qemu_mini_gguf.sh` | 已完成 Q2_K 路径接入 | Q3_K、Q6_K 和更多算子仍需扩展 |
| 功能验证与调优 | 提供一键验收脚本，自动构建、检查指令、运行逐函数验证、运行 benchmark 和 llama.cpp 冒烟测试 | `run_all_checks.sh`；`acceptance_reports/<时间戳>/` | 已完成 | 性能调优需要目标 CPU 数据 |
| 形成软件交付 | 发布包包含源码、二进制、静态库、模型、脚本、说明文档和校验文件 | `MANIFEST.md`；`README.md`；`SHA256SUMS` | 已完成 | 对外公开前需确认第三方模型/二进制许可证 |

## 三、中期计划对应表

| 中期计划 | 当前状态 | 证据 | 说明 |
| --- | --- | --- | --- |
| 从单一矩阵乘扩展到 Transformer 常用算子 | 已覆盖 Linear、MLP、SwiGLU、Self-Attention、Softmax、Norm、Embedding | `API_TEST_REPORT_MAPPING.md` | Attention 中 Q/K/V/O 投影走 SME packed Linear；QK^T 和 P*V 仍可继续优化 |
| 构建面向国产处理器的高性能计算库 | 已形成 `sme_ai_library` 静态库与公开头文件 | `sme_ai_library/README.md`；`examples/demo_linear.c` | 当前为展示型 AI 算子库，不是完整深度学习框架 |
| 构建代码模式库/文档 | 已补 SME1 指令说明、架构说明、测试方法、性能说明和局限说明 | `doc/`；`docs/` | 便于团队后续复用 |
| 与上层框架集成 | 已接入 llama.cpp Q2_K 路径 | `llama.cpp/doc/SME_Q2K_OPTIMIZATION.md` | PyTorch/ATen 自定义算子未纳入当前发布 |
| 论文/技术报告材料积累 | 已整理目标闭环、API 测试表、最终实验总结 | `GOAL_TRACEABILITY.md`；`FINAL_EXPERIMENT_SUMMARY.md` | 可直接转化为结题报告附录 |

## 四、当前完成度判断

| 维度 | 判断 |
| --- | --- |
| 软件交付完整性 | 已具备源码、库、二进制、脚本、模型、说明、校验和一键验收报告 |
| 功能正确性 | 逐函数验证覆盖全部公开 API，一键验收可自动生成报告 |
| SME 指令使用证据 | 函数库侧检查 `smstart/fmopa/smstop`，llama.cpp 侧检查 `smstart/smopa/smstop` |
| 应用接入 | 已接入 llama.cpp Q2_K 推理路径，并提供 REF/SME 对照 |
| 性能结论 | QEMU 只作为仿真参考；优秀结题若需强性能结论，应补目标 CPU 真机测试 |
| 编译器目标 | 已完成工具链和构建支持，未完成 GCC/LLVM 后端自动生成 SME 代码 |

## 五、答辩建议表述

建议主张：

> 本项目完成了面向 AArch64/SME1 的 AI 运算库和 llama.cpp Q2_K 推理路径接入，形成了可构建、可运行、可验证的软件发布包。项目以一键验收脚本、API-测试-报告对应表和最终实验总结保证成果可复现。

需要避免夸大：

> 当前 QEMU 数据不作为真实飞腾硬件性能结论；当前没有宣称已直接改造 GCC/LLVM 后端自动生成 SME 矩阵代码。
