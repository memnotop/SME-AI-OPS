# API 参考摘要

完整声明见 `sme_ai_library/include/sme_ai_ops.h`。API 与测试项的逐项对应关系见 `API_TEST_REPORT_MAPPING.md`。

## 一、环境查询

| API | 说明 |
| --- | --- |
| `sme_ai_has_sme` | 查询当前环境是否报告支持 SME |
| `sme_ai_has_sme2` | 查询当前环境是否报告支持 SME2；本项目不依赖 SME2 |
| `sme_ai_cntb` | 查询 SME vector length 字节数 |
| `sme_ai_cntw` | 查询 32-bit word lane 数量；当前微内核要求 `cntw == 16` |

## 二、生命周期管理

| API | 说明 |
| --- | --- |
| `sme_ai_workspace_init` | 初始化临时工作区 |
| `sme_ai_workspace_destroy` | 释放工作区 |
| `sme_ai_packed_linear_weight_init` | 初始化预打包 Linear 权重 |
| `sme_ai_packed_linear_weight_destroy` | 释放预打包权重 |

## 三、基础张量与激活

| 类别 | API |
| --- | --- |
| 基础张量 | `sme_ai_add_f32`、`sme_ai_mul_f32`、`sme_ai_residual_add_f32`、`sme_ai_add_bias_rowwise_f32` |
| 激活函数 | `sme_ai_relu_inplace_f32`、`sme_ai_sigmoid_inplace_f32`、`sme_ai_tanh_inplace_f32`、`sme_ai_gelu_inplace_f32`、`sme_ai_silu_inplace_f32` |
| Softmax | `sme_ai_softmax_rowwise_f32`、`sme_ai_softmax_masked_rowwise_f32`、`sme_ai_softmax_causal_rowwise_f32` |
| Norm | `sme_ai_layernorm_f32`、`sme_ai_rmsnorm_f32` |

## 四、高层 AI 算子

| 类别 | API |
| --- | --- |
| Embedding | `sme_ai_embedding_lookup_f16f32` |
| Linear | `sme_ai_linear_ref_f16f32`、`sme_ai_linear_sme_simple_f16f32`、`sme_ai_linear_sme_packed_f16f32` |
| Linear+Activation | `sme_ai_linear_relu_*`、`sme_ai_linear_gelu_*`、`sme_ai_linear_silu_*` |
| MLP | `sme_ai_mlp_relu_*`、`sme_ai_mlp_gelu_*` |
| SwiGLU | `sme_ai_swiglu_ref_f16f32`、`sme_ai_swiglu_sme_packed_f16f32` |
| Attention | `sme_ai_self_attention_ref_f16f32`、`sme_ai_self_attention_sme_packed_f16f32` |
| Conv/Pool | `sme_ai_conv2d_*`、`sme_ai_max_pool2d_nchw_f32`、`sme_ai_avg_pool2d_nchw_f32` |

## 五、调用建议

重复推理时建议使用：

1. `sme_ai_workspace_init`
2. `sme_ai_packed_linear_weight_init`
3. `sme_ai_linear_sme_packed_f16f32` 或基于 packed weight 的高层算子
4. `sme_ai_packed_linear_weight_destroy`
5. `sme_ai_workspace_destroy`

参考 `examples/demo_linear.c`。
