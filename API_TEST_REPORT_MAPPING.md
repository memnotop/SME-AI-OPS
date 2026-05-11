# API-测试-报告对应表

本文档用于终检验收时快速核对“公开 API 是否有测试、测试结果在哪里、是否纳入一键验收报告”。

## 一、核对口径

| 项目 | 说明 |
| --- | --- |
| 公开 API 来源 | `sme_ai_library/include/sme_ai_ops.h` |
| 逐函数验证程序 | `sme_ai_library/build/verify_sme_ai_timing` |
| 逐函数验证源码 | `sme_ai_library/test_bench/verify_sme_ai_timing.c` |
| 逐函数验证脚本 | `sme_ai_library/run_qemu_verify_timing.sh` |
| 性能对比程序 | `sme_ai_library/build/bench_sme_full_timing` |
| 性能对比源码 | `sme_ai_library/test_bench/bench_sme_full_timing.c` |
| 性能对比脚本 | `sme_ai_library/run_qemu_full_bench.sh` |
| 一键验收脚本 | `run_all_checks.sh` |
| 一键验收报告目录 | `acceptance_reports/<时间戳>/acceptance_report.md` |
| 逐函数验证日志 | 一键报告中的“SME1 AI 运算库逐函数 QEMU 验证”步骤 |
| benchmark 日志 | 一键报告中的“SME1 AI 运算库 REF/SME QEMU benchmark”步骤 |

## 二、生成验收报告

在发布包根目录执行：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
./run_all_checks.sh
```

脚本成功后会输出类似：

```text
验收报告：/home/liumingjian/dachuang/SME-AI-OPS-v1.0/acceptance_reports/<时间戳>/acceptance_report.md
日志目录：/home/liumingjian/dachuang/SME-AI-OPS-v1.0/acceptance_reports/<时间戳>/logs
```

验收时优先看报告中的三类步骤：

| 报告步骤 | 验收意义 |
| --- | --- |
| `SME1 AI 运算库重新构建` | 证明源码可以重新生成静态库和测试程序 |
| `SME1 AI 运算库指令检查` | 证明函数库二进制中包含 `smstart/fmopa/smstop` |
| `SME1 AI 运算库逐函数 QEMU 验证` | 证明公开 API 均能运行并输出 `OK` |
| `SME1 AI 运算库 REF/SME QEMU benchmark` | 证明高层算子可做 REF/SME 路径对比 |

## 三、逐函数 API 对应表

说明：

- “逐函数验证用例”对应 `run_qemu_verify_timing.sh` 中的 `--case` 名称。
- “benchmark 用例”对应 `run_qemu_full_bench.sh` 中的 `--case` 名称。
- “无独立 benchmark”表示该 API 已在逐函数验证中覆盖，但不作为单独性能对比项。
- 对于 `workspace` 和 `packed weight` 生命周期 API，部分高层 benchmark 也会间接调用它们。

| API | 功能类别 | 逐函数验证用例 | benchmark 用例 | 报告核对位置 |
| --- | --- | --- | --- | --- |
| `sme_ai_has_sme` | SME 环境查询 | `sme_ai_has_sme` | 高层 benchmark 启动时间接检查 | 逐函数验证日志；benchmark 日志开头 |
| `sme_ai_has_sme2` | SME 环境查询 | `sme_ai_has_sme2` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_cntb` | SME 向量长度查询 | `sme_ai_cntb` | 高层 benchmark 启动时须满足环境要求 | 逐函数验证日志；benchmark 日志开头 |
| `sme_ai_cntw` | SME 向量长度查询 | `sme_ai_cntw` | 高层 benchmark 启动时须满足 `cntw == 16` | 逐函数验证日志；benchmark 日志开头 |
| `sme_ai_workspace_init` | 工作区生命周期 | `sme_ai_workspace_init` | `linear_*`、`mlp_*`、`swiglu`、`attention`、`conv2d` 间接覆盖 | 逐函数验证日志；benchmark 日志 |
| `sme_ai_workspace_destroy` | 工作区生命周期 | `sme_ai_workspace_destroy` | `linear_*`、`mlp_*`、`swiglu`、`attention`、`conv2d` 间接覆盖 | 逐函数验证日志；benchmark 日志 |
| `sme_ai_packed_linear_weight_init` | 预打包权重生命周期 | `sme_ai_packed_linear_weight_init` | `linear_packed`、`linear_relu`、`linear_gelu`、`linear_silu`、`mlp_*`、`swiglu`、`attention` 间接覆盖 | 逐函数验证日志；benchmark 日志 |
| `sme_ai_packed_linear_weight_destroy` | 预打包权重生命周期 | `sme_ai_packed_linear_weight_destroy` | `linear_packed`、`linear_relu`、`linear_gelu`、`linear_silu`、`mlp_*`、`swiglu`、`attention` 间接覆盖 | 逐函数验证日志；benchmark 日志 |
| `sme_ai_add_f32` | 基础张量算子 | `sme_ai_add_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_mul_f32` | 基础张量算子 | `sme_ai_mul_f32` | `swiglu` 间接覆盖门控乘法 | 逐函数验证日志；`swiglu` benchmark 日志 |
| `sme_ai_residual_add_f32` | 基础张量算子 | `sme_ai_residual_add_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_add_bias_rowwise_f32` | 基础张量算子 | `sme_ai_add_bias_rowwise_f32` | Linear/MLP/Attention/Conv2D 中的 bias 路径间接覆盖 | 逐函数验证日志；benchmark 日志 |
| `sme_ai_relu_inplace_f32` | 激活函数 | `sme_ai_relu_inplace_f32` | `linear_relu`、`mlp_relu` 间接覆盖 | 逐函数验证日志；`linear_relu`、`mlp_relu` benchmark 日志 |
| `sme_ai_sigmoid_inplace_f32` | 激活函数 | `sme_ai_sigmoid_inplace_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_tanh_inplace_f32` | 激活函数 | `sme_ai_tanh_inplace_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_gelu_inplace_f32` | 激活函数 | `sme_ai_gelu_inplace_f32` | `linear_gelu`、`mlp_gelu` 间接覆盖 | 逐函数验证日志；`linear_gelu`、`mlp_gelu` benchmark 日志 |
| `sme_ai_silu_inplace_f32` | 激活函数 | `sme_ai_silu_inplace_f32` | `linear_silu`、`swiglu` 间接覆盖 | 逐函数验证日志；`linear_silu`、`swiglu` benchmark 日志 |
| `sme_ai_softmax_rowwise_f32` | Softmax | `sme_ai_softmax_rowwise_f32` | `attention` 间接覆盖 | 逐函数验证日志；`attention` benchmark 日志 |
| `sme_ai_softmax_masked_rowwise_f32` | Softmax | `sme_ai_softmax_masked_rowwise_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_softmax_causal_rowwise_f32` | Softmax | `sme_ai_softmax_causal_rowwise_f32` | `attention` causal 路径间接覆盖 | 逐函数验证日志；`attention` benchmark 日志 |
| `sme_ai_layernorm_f32` | 归一化 | `sme_ai_layernorm_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_rmsnorm_f32` | 归一化 | `sme_ai_rmsnorm_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_embedding_lookup_f16f32` | Embedding | `sme_ai_embedding_lookup_f16f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_max_pool2d_nchw_f32` | Pooling | `sme_ai_max_pool2d_nchw_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_avg_pool2d_nchw_f32` | Pooling | `sme_ai_avg_pool2d_nchw_f32` | 无独立 benchmark | 逐函数验证日志 |
| `sme_ai_linear_ref_f16f32` | Linear 参考实现 | `sme_ai_linear_ref_f16f32` | `linear_simple`、`linear_packed` 的 REF 路径 | 逐函数验证日志；`linear_simple`、`linear_packed` benchmark 日志 |
| `sme_ai_linear_sme_simple_f16f32` | Linear SME 实现 | `sme_ai_linear_sme_simple_f16f32` | `linear_simple` 的 SME 路径 | 逐函数验证日志；`linear_simple` benchmark 日志 |
| `sme_ai_linear_sme_packed_f16f32` | Linear SME packed 实现 | `sme_ai_linear_sme_packed_f16f32` | `linear_packed` 的 SME 路径 | 逐函数验证日志；`linear_packed` benchmark 日志 |
| `sme_ai_linear_relu_ref_f16f32` | Linear+ReLU 参考实现 | `sme_ai_linear_relu_ref_f16f32` | `linear_relu` 的 REF 路径 | 逐函数验证日志；`linear_relu` benchmark 日志 |
| `sme_ai_linear_relu_sme_packed_f16f32` | Linear+ReLU SME packed 实现 | `sme_ai_linear_relu_sme_packed_f16f32` | `linear_relu` 的 SME 路径 | 逐函数验证日志；`linear_relu` benchmark 日志 |
| `sme_ai_linear_gelu_ref_f16f32` | Linear+GELU 参考实现 | `sme_ai_linear_gelu_ref_f16f32` | `linear_gelu` 的 REF 路径 | 逐函数验证日志；`linear_gelu` benchmark 日志 |
| `sme_ai_linear_gelu_sme_packed_f16f32` | Linear+GELU SME packed 实现 | `sme_ai_linear_gelu_sme_packed_f16f32` | `linear_gelu` 的 SME 路径 | 逐函数验证日志；`linear_gelu` benchmark 日志 |
| `sme_ai_linear_silu_ref_f16f32` | Linear+SiLU 参考实现 | `sme_ai_linear_silu_ref_f16f32` | `linear_silu` 的 REF 路径 | 逐函数验证日志；`linear_silu` benchmark 日志 |
| `sme_ai_linear_silu_sme_packed_f16f32` | Linear+SiLU SME packed 实现 | `sme_ai_linear_silu_sme_packed_f16f32` | `linear_silu` 的 SME 路径 | 逐函数验证日志；`linear_silu` benchmark 日志 |
| `sme_ai_mlp_relu_ref_f16f32` | MLP+ReLU 参考实现 | `sme_ai_mlp_relu_ref_f16f32` | `mlp_relu` 的 REF 路径 | 逐函数验证日志；`mlp_relu` benchmark 日志 |
| `sme_ai_mlp_relu_sme_packed_f16f32` | MLP+ReLU SME packed 实现 | `sme_ai_mlp_relu_sme_packed_f16f32` | `mlp_relu` 的 SME 路径 | 逐函数验证日志；`mlp_relu` benchmark 日志 |
| `sme_ai_mlp_gelu_ref_f16f32` | MLP+GELU 参考实现 | `sme_ai_mlp_gelu_ref_f16f32` | `mlp_gelu` 的 REF 路径 | 逐函数验证日志；`mlp_gelu` benchmark 日志 |
| `sme_ai_mlp_gelu_sme_packed_f16f32` | MLP+GELU SME packed 实现 | `sme_ai_mlp_gelu_sme_packed_f16f32` | `mlp_gelu` 的 SME 路径 | 逐函数验证日志；`mlp_gelu` benchmark 日志 |
| `sme_ai_swiglu_ref_f16f32` | SwiGLU 参考实现 | `sme_ai_swiglu_ref_f16f32` | `swiglu` 的 REF 路径 | 逐函数验证日志；`swiglu` benchmark 日志 |
| `sme_ai_swiglu_sme_packed_f16f32` | SwiGLU SME packed 实现 | `sme_ai_swiglu_sme_packed_f16f32` | `swiglu` 的 SME 路径 | 逐函数验证日志；`swiglu` benchmark 日志 |
| `sme_ai_self_attention_ref_f16f32` | Self-Attention 参考实现 | `sme_ai_self_attention_ref_f16f32` | `attention` 的 REF 路径 | 逐函数验证日志；`attention` benchmark 日志 |
| `sme_ai_self_attention_sme_packed_f16f32` | Self-Attention SME packed 实现 | `sme_ai_self_attention_sme_packed_f16f32` | `attention` 的 SME 路径 | 逐函数验证日志；`attention` benchmark 日志 |
| `sme_ai_conv2d_ref_nchw_f16f32` | Conv2D 参考实现 | `sme_ai_conv2d_ref_nchw_f16f32` | `conv2d` 的 REF 路径 | 逐函数验证日志；`conv2d` benchmark 日志 |
| `sme_ai_conv2d_sme_im2col_nchw_f16f32` | Conv2D SME im2col 实现 | `sme_ai_conv2d_sme_im2col_nchw_f16f32` | `conv2d` 的 SME 路径 | 逐函数验证日志；`conv2d` benchmark 日志 |

## 四、benchmark 用例对应表

| benchmark 用例 | 覆盖的主要 API | 输出字段 | 备注 |
| --- | --- | --- | --- |
| `linear_simple` | `sme_ai_linear_ref_f16f32`、`sme_ai_linear_sme_simple_f16f32` | `REF_ms`、`SME_ms`、`SPEEDUP`、`CHECK`、`MAX_ABS`、`MAX_REL`、`CHECKSUM` | 验证直接传入原始权重的 Linear SME 路径 |
| `linear_packed` | `sme_ai_linear_ref_f16f32`、`sme_ai_linear_sme_packed_f16f32`、packed weight 生命周期 API | 同上 | 验证预打包权重 Linear 路径 |
| `linear_relu` | `sme_ai_linear_relu_ref_f16f32`、`sme_ai_linear_relu_sme_packed_f16f32` | 同上 | 验证 Linear+ReLU |
| `linear_gelu` | `sme_ai_linear_gelu_ref_f16f32`、`sme_ai_linear_gelu_sme_packed_f16f32` | 同上 | 验证 Transformer 常见 GELU 激活 |
| `linear_silu` | `sme_ai_linear_silu_ref_f16f32`、`sme_ai_linear_silu_sme_packed_f16f32` | 同上 | 验证 LLaMA/SwiGLU 常见 SiLU 激活 |
| `mlp_relu` | `sme_ai_mlp_relu_ref_f16f32`、`sme_ai_mlp_relu_sme_packed_f16f32` | 同上 | 验证两层 MLP+ReLU |
| `mlp_gelu` | `sme_ai_mlp_gelu_ref_f16f32`、`sme_ai_mlp_gelu_sme_packed_f16f32` | 同上 | 验证两层 MLP+GELU |
| `swiglu` | `sme_ai_swiglu_ref_f16f32`、`sme_ai_swiglu_sme_packed_f16f32`、`sme_ai_mul_f32`、`sme_ai_silu_inplace_f32` | 同上 | 验证 LLaMA 类 gated FFN |
| `attention` | `sme_ai_self_attention_ref_f16f32`、`sme_ai_self_attention_sme_packed_f16f32`、softmax 相关 API | 同上 | 验证 Q/K/V/O 投影和 causal attention 调用链 |
| `conv2d` | `sme_ai_conv2d_ref_nchw_f16f32`、`sme_ai_conv2d_sme_im2col_nchw_f16f32` | 同上 | 验证 im2col+SME packed Linear 的卷积路径 |

## 五、验收结论写法建议

可在终检报告或答辩材料中使用如下表述：

> 本发布包已建立公开 API、测试用例和验收报告之间的对应关系。`run_all_checks.sh` 会自动重新构建 SME1 AI 运算库，并通过 `verify_sme_ai_timing` 对 `sme_ai_ops.h` 中公开 API 逐项验证；同时通过 `bench_sme_full_timing` 对 Linear、MLP、SwiGLU、Attention、Conv2D 等高层算子进行 REF/SME 路径对比。所有验证输出统一归档到 `acceptance_reports/<时间戳>/`，便于现场复现和逐项核查。

同时需要保留如下边界说明：

> QEMU 验证用于证明程序可运行、数值检查可通过和 SME 调用链已打通；真实硬件速度提升仍需以支持 SME1 的目标 CPU 实测为准。
