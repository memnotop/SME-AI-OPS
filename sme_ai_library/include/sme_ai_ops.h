#ifndef SME_AI_OPS_H
#define SME_AI_OPS_H

#include <stdint.h>

/*
 * sme_ai_ops.h
 *
 * 这个头文件是 SME1 AI 运算库对外暴露的“说明书”。
 *
 * 新手阅读建议：
 * 1. 先看 sme_ai_workspace：它是工作区，用来提前申请中间缓冲区。
 * 2. 再看 sme_ai_packed_linear_weight：它是预打包权重，用来加速重复执行的 Linear。
 * 3. 函数名中带 ref 的是参考实现，主要用于校验正确性，不使用 SME 微内核。
 * 4. 函数名中带 sme 的是 SME1 实现，主要使用 16x16 FMOPA GEMM 微内核。
 * 5. 函数名中带 packed 的表示权重已经被预先整理成 SME 更容易读取的布局。
 * 6. 函数名中带 f16f32 的含义是：输入/权重常用 FP16，累加和输出使用 FP32。
 *
 * 常见张量形状约定：
 * - Linear input:  [batch, in_features]
 * - Linear weight: [in_features, out_features]
 * - Linear output: [batch, out_features]
 * - NCHW:          [batch, channels, height, width]
 *
 * 注意：
 * - 本库面向 SME1 目标 CPU。当前 SME GEMM 微内核要求 cntw == 16，
 *   即 512-bit SME vector length。
 * - 本库不是完整深度学习框架，而是一组便于展示 SME1 加速效果的基础 AI 算子。
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 所有可能返回错误的函数都使用 sme_ai_status。
 * 返回 SME_AI_OK 表示成功；负数表示失败原因。
 */
typedef enum {
    SME_AI_OK = 0,                    /* 成功。 */
    SME_AI_BAD_ARGUMENT = -1,         /* 参数错误，例如传入 NULL、维度为 0、形状不匹配。 */
    SME_AI_ALLOC_FAILED = -2,         /* 内存申请失败。 */
    SME_AI_UNSUPPORTED = -3,          /* 当前 CPU 或向量长度不支持本实现。 */
    SME_AI_WORKSPACE_TOO_SMALL = -4,  /* 工作区容量不够，需要用更大的 max_* 重新初始化。 */
} sme_ai_status;

/*
 * SME AI 工作区。
 *
 * 为什么需要工作区？
 * - AI 算子中经常需要临时数组，例如：
 *   - GEMM 的 A tile 打包缓冲区；
 *   - MLP 第一层输出；
 *   - Attention 的 Q/K/V、score、context；
 *   - Conv2D 的 im2col 临时矩阵。
 * - 如果每次函数调用都 malloc/free，会让测试时间受到内存分配干扰。
 * - 因此本库在初始化时一次性申请缓冲区，后续函数重复使用。
 *
 * max_batch / max_in_features / max_out_features 决定这个工作区能处理的最大规模。
 */
typedef struct {
    uint64_t max_batch;          /* 最大 batch 或最大序列长度。Attention 中 seq_len 也使用这个限制。 */
    uint64_t max_in_features;    /* Linear/GEMM 输入维度上限。 */
    uint64_t max_out_features;   /* Linear/GEMM 输出维度上限。 */
    uint64_t cntb;               /* SME vector length 的字节数。512-bit 时 cntb = 64。 */
    uint64_t cntw;               /* 一个 SME 向量中 32-bit word 的数量。512-bit 时 cntw = 16。 */
    uint64_t max_h;              /* 预留字段，当前等于 max_in_features，用于描述最大隐藏维度。 */
    uint64_t max_m_pad;          /* batch 方向按 16 对齐后的大小。 */
    uint64_t max_k_pad;          /* in_features 方向按 16 对齐后的大小。 */
    uint64_t max_n_pad;          /* out_features 方向按 16 对齐后的大小。 */
    float *a_pairs;              /* 通用 FP32 临时缓冲区，Attention 中保存 context。 */
    float *a_mod;                /* GEMM 的 A tile 打包缓冲区，布局为 [k_pad, 16]。 */
    float *b_pack;               /* 临时权重打包缓冲区，用于 simple 版本 Linear。 */
    float *tmp_f32;              /* 第一个通用 FP32 临时缓冲区。 */
    float *tmp2_f32;             /* 第二个通用 FP32 临时缓冲区。 */
    __fp16 *tmp_f16;             /* FP32 中间结果转回 FP16 后保存于此，供下一层 Linear 使用。 */
    float *gemm_c_tile;          /* 边界 tile 临时输出，大小为 16x16。 */
    float *attn_scores;          /* Attention 的 score 矩阵，形状通常为 [seq_len, seq_len]。 */
} sme_ai_workspace;

/*
 * 预打包 Linear 权重。
 *
 * 原始 Linear 权重形状是 [in_features, out_features]。
 * SME1 16x16 GEMM 微内核更喜欢按 16 列 tile 存放权重：
 * - k_pad: in_features 按 16 补齐。
 * - n_pad: out_features 按 16 补齐。
 * - packed_weight: 已经转换成 float 并按 tile 排列的权重。
 *
 * 对同一层权重重复推理时，应优先使用 packed 版本，避免每次运行都重新打包。
 */
typedef struct {
    uint64_t in_features;     /* 原始输入维度。 */
    uint64_t out_features;    /* 原始输出维度。 */
    uint64_t k_pad;           /* 输入维度按 16 对齐后的大小。 */
    uint64_t n_pad;           /* 输出维度按 16 对齐后的大小。 */
    float *packed_weight;     /* 预打包权重数据。使用完必须调用 destroy 释放。 */
} sme_ai_packed_linear_weight;

/* -------- CPU/SME 环境查询 -------- */

/* 返回当前运行环境是否报告支持 SME。 */
int sme_ai_has_sme(void);

/* 返回当前运行环境是否报告支持 SME2。本交付不依赖 SME2，只用于打印环境信息。 */
int sme_ai_has_sme2(void);

/* 返回 SME vector length 的字节数。目标 512-bit 时应为 64。 */
uint64_t sme_ai_cntb(void);

/* 返回一个向量中 32-bit 元素数量。目标 512-bit 时应为 16。 */
uint64_t sme_ai_cntw(void);

/* -------- 工作区与权重生命周期管理 -------- */

/*
 * 初始化工作区。
 *
 * max_batch/max_in_features/max_out_features 要覆盖后续所有测试规模。
 * 如果后续函数返回 SME_AI_WORKSPACE_TOO_SMALL，通常说明这里给的 max_* 太小。
 */
sme_ai_status sme_ai_workspace_init(sme_ai_workspace *ws, uint64_t max_batch, uint64_t max_in_features, uint64_t max_out_features);

/* 释放工作区内部申请的内存。调用后 ws 会被清零。 */
void sme_ai_workspace_destroy(sme_ai_workspace *ws);

/*
 * 初始化预打包 Linear 权重。
 *
 * weight 原始布局为 [in_features, out_features]，元素类型为 FP16。
 * packed 初始化后可以在多个 batch 上重复使用。
 */
sme_ai_status sme_ai_packed_linear_weight_init(
    sme_ai_packed_linear_weight *packed,
    const __fp16 *weight,
    uint64_t in_features,
    uint64_t out_features);

/* 释放预打包权重。 */
void sme_ai_packed_linear_weight_destroy(sme_ai_packed_linear_weight *packed);

/* -------- Linear / 全连接层 -------- */

/*
 * 参考 Linear：
 * output = input x weight + bias
 *
 * 主要用于和 SME 版本做正确性对比。实现简单、容易理解，但速度不是重点。
 */
sme_ai_status sme_ai_linear_ref_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output);

/*
 * SME simple Linear。
 *
 * 调用时直接传入原始 FP16 weight，函数内部会临时打包到 ws->b_pack。
 * 适合演示或偶尔调用；如果同一层权重会重复使用，建议用 packed 版本。
 */
sme_ai_status sme_ai_linear_sme_simple_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output);

/*
 * SME packed Linear。
 *
 * 使用 sme_ai_packed_linear_weight_init 预打包过的权重。
 * 推理时推荐使用该版本，因为权重打包成本只发生一次。
 */
sme_ai_status sme_ai_linear_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output);

/* Linear + ReLU。ReLU 是 max(x, 0)，常用于 MLP。 */
sme_ai_status sme_ai_linear_relu_ref_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output);

/* SME packed Linear + ReLU。 */
sme_ai_status sme_ai_linear_relu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output);

/* -------- MLP / 前馈网络 -------- */

/*
 * 两层 MLP + ReLU：
 * hidden = ReLU(input x fc1_weight + fc1_bias)
 * output = hidden x fc2_weight + fc2_bias
 */
sme_ai_status sme_ai_mlp_relu_ref_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *fc1_weight,
    uint64_t hidden_features,
    const float *fc1_bias,
    const __fp16 *fc2_weight,
    uint64_t out_features,
    const float *fc2_bias,
    float *output);

/* SME packed 两层 MLP + ReLU。 */
sme_ai_status sme_ai_mlp_relu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *fc1_weight,
    const float *fc1_bias,
    const sme_ai_packed_linear_weight *fc2_weight,
    const float *fc2_bias,
    float *output);

/* -------- 基础逐元素算子和常用激活函数 -------- */

/* output[i] = lhs[i] + rhs[i]。用于张量加法。 */
void sme_ai_add_f32(float *output, const float *lhs, const float *rhs, uint64_t count);

/* output[i] = lhs[i] * rhs[i]。用于门控结构、逐元素缩放。 */
void sme_ai_mul_f32(float *output, const float *lhs, const float *rhs, uint64_t count);

/* output = input + residual。Transformer 残差连接常用。 */
void sme_ai_residual_add_f32(float *output, const float *input, const float *residual, uint64_t count);

/* 对矩阵每一行加同一个 bias 向量：output[row, col] = input[row, col] + bias[col]。 */
void sme_ai_add_bias_rowwise_f32(float *output, const float *input, const float *bias, uint64_t rows, uint64_t cols);

/* 原地 ReLU：data[i] = max(data[i], 0)。 */
void sme_ai_relu_inplace_f32(float *data, uint64_t count);

/* 原地 sigmoid：data[i] = 1 / (1 + exp(-data[i]))。 */
void sme_ai_sigmoid_inplace_f32(float *data, uint64_t count);

/* 原地 tanh。 */
void sme_ai_tanh_inplace_f32(float *data, uint64_t count);

/* 原地 GELU，Transformer 前馈层常用激活。 */
void sme_ai_gelu_inplace_f32(float *data, uint64_t count);

/* 原地 SiLU/Swish，LLaMA/SwiGLU 中常用。 */
void sme_ai_silu_inplace_f32(float *data, uint64_t count);

/* 对每一行做 softmax，常用于 attention score 归一化。 */
void sme_ai_softmax_rowwise_f32(float *output, const float *input, uint64_t rows, uint64_t cols);

/* 带 mask 的 softmax。mask 通常包含 0 或一个很大的负数，用于屏蔽非法位置。 */
void sme_ai_softmax_masked_rowwise_f32(float *output, const float *input, const float *mask, uint64_t rows, uint64_t cols);

/* causal softmax：第 i 行只能看第 0..i 列，用于自回归语言模型。 */
void sme_ai_softmax_causal_rowwise_f32(float *output, const float *input, uint64_t rows, uint64_t cols);

/* LayerNorm：按行计算均值和方差，再做归一化，可选 gamma/beta。 */
void sme_ai_layernorm_f32(
    float *output,
    const float *input,
    const float *gamma,
    const float *beta,
    uint64_t rows,
    uint64_t cols,
    float eps);

/* RMSNorm：只使用均方根进行归一化，LLaMA 类模型常用。 */
void sme_ai_rmsnorm_f32(
    float *output,
    const float *input,
    const float *weight,
    uint64_t rows,
    uint64_t cols,
    float eps);

/*
 * Embedding lookup。
 *
 * 根据 token_ids 从 embedding table 中取出向量。
 * table 布局为 [vocab_size, hidden_size]，输出为 [num_tokens, hidden_size]。
 */
sme_ai_status sme_ai_embedding_lookup_f16f32(
    float *output,
    const __fp16 *table,
    const uint32_t *token_ids,
    uint64_t num_tokens,
    uint64_t vocab_size,
    uint64_t hidden_size);

/* Linear + GELU。 */
sme_ai_status sme_ai_linear_gelu_ref_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output);

/* SME packed Linear + GELU。 */
sme_ai_status sme_ai_linear_gelu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output);

/* Linear + SiLU。 */
sme_ai_status sme_ai_linear_silu_ref_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output);

/* SME packed Linear + SiLU。 */
sme_ai_status sme_ai_linear_silu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output);

/* 两层 MLP + GELU。 */
sme_ai_status sme_ai_mlp_gelu_ref_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *fc1_weight,
    uint64_t hidden_features,
    const float *fc1_bias,
    const __fp16 *fc2_weight,
    uint64_t out_features,
    const float *fc2_bias,
    float *output);

/* SME packed 两层 MLP + GELU。 */
sme_ai_status sme_ai_mlp_gelu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *fc1_weight,
    const float *fc1_bias,
    const sme_ai_packed_linear_weight *fc2_weight,
    const float *fc2_bias,
    float *output);

/*
 * SwiGLU 参考实现。
 *
 * LLaMA 前馈层常见结构：
 * gate = SiLU(input x gate_weight + gate_bias)
 * up   = input x up_weight + up_bias
 * hid  = gate * up
 * out  = hid x down_weight + down_bias
 */
sme_ai_status sme_ai_swiglu_ref_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *gate_weight,
    uint64_t hidden_features,
    const float *gate_bias,
    const __fp16 *up_weight,
    const float *up_bias,
    const __fp16 *down_weight,
    uint64_t out_features,
    const float *down_bias,
    float *output);

/* SME packed SwiGLU。 */
sme_ai_status sme_ai_swiglu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *gate_weight,
    const float *gate_bias,
    const sme_ai_packed_linear_weight *up_weight,
    const float *up_bias,
    const sme_ai_packed_linear_weight *down_weight,
    const float *down_bias,
    float *output);

/*
 * Self-Attention 参考实现。
 *
 * 计算流程：
 * Q = input x q_weight
 * K = input x k_weight
 * V = input x v_weight
 * score = softmax(Q x K^T / sqrt(head_dim))
 * context = score x V
 * output = context x out_weight
 */
sme_ai_status sme_ai_self_attention_ref_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t seq_len,
    uint64_t model_dim,
    uint64_t num_heads,
    const __fp16 *q_weight,
    const float *q_bias,
    const __fp16 *k_weight,
    const float *k_bias,
    const __fp16 *v_weight,
    const float *v_bias,
    const __fp16 *out_weight,
    const float *out_bias,
    int causal,
    float *output);

/* SME packed Self-Attention：Q/K/V/O 四个投影使用 SME packed Linear。 */
sme_ai_status sme_ai_self_attention_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t seq_len,
    uint64_t model_dim,
    uint64_t num_heads,
    const sme_ai_packed_linear_weight *q_weight,
    const float *q_bias,
    const sme_ai_packed_linear_weight *k_weight,
    const float *k_bias,
    const sme_ai_packed_linear_weight *v_weight,
    const float *v_bias,
    const sme_ai_packed_linear_weight *out_weight,
    const float *out_bias,
    int causal,
    float *output);

/*
 * 参考 Conv2D。
 *
 * 输入和输出均为 NCHW 布局：
 * input:  [batch, in_channels, in_h, in_w]
 * weight: [out_channels, in_channels, kernel_h, kernel_w]
 * output: [batch, out_channels, out_h, out_w]
 */
sme_ai_status sme_ai_conv2d_ref_nchw_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_channels,
    uint64_t in_h,
    uint64_t in_w,
    const __fp16 *weight,
    uint64_t out_channels,
    uint64_t kernel_h,
    uint64_t kernel_w,
    uint64_t stride_h,
    uint64_t stride_w,
    uint64_t pad_h,
    uint64_t pad_w,
    const float *bias,
    float *output);

/*
 * SME Conv2D。
 *
 * 使用 im2col 把卷积转换成矩阵乘：
 * - 每个输出位置展开成一行 patch。
 * - 卷积核展开成 Linear 权重。
 * - 最后调用 SME packed Linear。
 */
sme_ai_status sme_ai_conv2d_sme_im2col_nchw_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_channels,
    uint64_t in_h,
    uint64_t in_w,
    const __fp16 *weight,
    uint64_t out_channels,
    uint64_t kernel_h,
    uint64_t kernel_w,
    uint64_t stride_h,
    uint64_t stride_w,
    uint64_t pad_h,
    uint64_t pad_w,
    const float *bias,
    float *output);

/* NCHW 2D max pooling。 */
void sme_ai_max_pool2d_nchw_f32(
    float *output,
    const float *input,
    uint64_t batch,
    uint64_t channels,
    uint64_t in_h,
    uint64_t in_w,
    uint64_t kernel_h,
    uint64_t kernel_w,
    uint64_t stride_h,
    uint64_t stride_w);

/* NCHW 2D average pooling。 */
void sme_ai_avg_pool2d_nchw_f32(
    float *output,
    const float *input,
    uint64_t batch,
    uint64_t channels,
    uint64_t in_h,
    uint64_t in_w,
    uint64_t kernel_h,
    uint64_t kernel_w,
    uint64_t stride_h,
    uint64_t stride_w);

#ifdef __cplusplus
}
#endif

#endif
