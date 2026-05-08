#define _GNU_SOURCE

#include "sme_ai_ops.h"

#include <arm_sve.h>
#include <asm/hwcap.h>
#include <linux/prctl.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

/*
 * sme_ai_ops.c
 *
 * 这个文件是 SME1 AI 运算库的主体实现。为了方便新手阅读，先说明几个核心概念：
 *
 * 1. Linear / GEMM 是本库的核心
 *    很多 AI 算子最终都能变成矩阵乘：
 *    - Linear 本身就是矩阵乘。
 *    - MLP 是两个 Linear 中间夹一个激活函数。
 *    - SwiGLU 是多个 Linear 加门控乘法。
 *    - Attention 中的 Q/K/V/O 投影都是 Linear。
 *    - Conv2D 可以通过 im2col 转换成 Linear/GEMM。
 *
 * 2. SME1 微内核负责加速 16x16 GEMM tile
 *    本库的 SME 汇编文件 sme1_gemm_tile_16x16.S 使用 FMOPA 指令。
 *    一次 FMOPA 可以把 16 个 A 元素和 16 个 B 元素做外积，
 *    并累加到 SME 的 ZA 矩阵累加器中。
 *
 * 3. 为什么要打包 packed weight
 *    原始权重通常按 [in_features, out_features] 存放。
 *    SME 微内核希望每次读取连续的 16 个输出列，形成 16x16 tile。
 *    所以我们预先把权重整理成更适合 SME 读取的布局，避免每次推理都重新整理。
 *
 * 4. 为什么大量函数使用 workspace
 *    workspace 是提前申请好的临时内存池。
 *    这样 benchmark 计时主要测算子本身，而不是反复 malloc/free 的开销。
 *
 * 5. ref 和 sme 的区别
 *    - ref：参考实现，结构简单，用于校验结果。
 *    - sme：SME1 实现，走 16x16 FMOPA 微内核，用于展示加速。
 */

/*
 * 这两个函数在 sme1_gemm_tile_16x16.S 中实现。
 * C 代码负责准备输入数据和调用它们；真正的 SME FMOPA 指令在汇编里。
 */
extern void sme1_gemm_tile_16x16_k(const float *a_packed, const float *b_packed, float *c_tile, uint64_t k);
extern void sme1_gemm_tile_16x16_strided_k(
    const float *a_packed,
    const float *b_packed,
    float *c_tile,
    uint64_t k,
    uint64_t row_stride_bytes);

typedef enum {
    SME_AI_ACT_NONE = 0,
    SME_AI_ACT_RELU,
    SME_AI_ACT_GELU,
    SME_AI_ACT_SILU,
} sme_ai_activation_kind;

enum {
    /*
     * 当前 SME1 GEMM 微内核固定使用 16x16 tile。
     * 当 SME 向量长度为 512 bit 时，一个向量可以放 16 个 FP32，
     * 所以 tile 的一条边自然是 16。
     */
    SME1_TILE = 16,
};

/* 申请指定 alignment 对齐的内存，并清零。SME/SVE 读取连续对齐数据更友好。 */
static void *aligned_calloc_bytes(size_t alignment, size_t bytes) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        return NULL;
    }
    memset(ptr, 0, bytes);
    return ptr;
}

/* 把 value 向上补齐到 align 的整数倍。例如 round_up_u64(17, 16) = 32。 */
static uint64_t round_up_u64(uint64_t value, uint64_t align) {
    return (value + align - 1) / align * align;
}

/*
 * 根据卷积输入尺寸、卷积核、步长和 padding 计算输出高/宽。
 * 公式：out = floor((in + 2 * pad - kernel) / stride) + 1。
 */
static uint64_t conv_out_dim(uint64_t in_size, uint64_t kernel, uint64_t stride, uint64_t pad) {
    uint64_t padded = in_size + 2 * pad;
    if (kernel == 0 || stride == 0 || padded < kernel) {
        return 0;
    }
    return (padded - kernel) / stride + 1;
}

/*
 * GELU 近似公式。
 * GELU 是 Transformer 前馈层中很常见的激活函数，比 ReLU 更平滑。
 */
static float gelu_approx(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x3)));
}

/*
 * 读取当前 SME vector length 的字节数。
 *
 * prctl(PR_SME_GET_VL) 是 Linux 上查询 SME VL 的方式。
 * 如果查询失败，后面的 sme_ai_cntb/sme_ai_cntw 会退回到汇编 cntb/cntw。
 */
static uint64_t read_sme_vl_bytes(void) {
    long sme_vl = prctl(PR_SME_GET_VL);
    if (sme_vl >= 0) {
        return (uint64_t) sme_vl & PR_SME_VL_LEN_MASK;
    }
    return 0;
}

/* cntb 表示当前向量长度有多少字节。512-bit SME 时 cntb = 64。 */
uint64_t sme_ai_cntb(void) {
    uint64_t sme_vl_bytes = read_sme_vl_bytes();
    if (sme_vl_bytes != 0) {
        return sme_vl_bytes;
    }
    {
        uint64_t value = 0;
        __asm__ volatile("cntb %0" : "=r"(value));
        return value;
    }
}

/* cntw 表示当前向量长度有多少个 32-bit word。512-bit SME 时 cntw = 16。 */
uint64_t sme_ai_cntw(void) {
    uint64_t sme_vl_bytes = read_sme_vl_bytes();
    if (sme_vl_bytes != 0) {
        return sme_vl_bytes / sizeof(float);
    }
    {
        uint64_t value = 0;
        __asm__ volatile("cntw %0" : "=r"(value));
        return value;
    }
}

/* 从 Linux HWCAP2 中判断 CPU/内核是否报告支持 SME。 */
int sme_ai_has_sme(void) {
    return (getauxval(AT_HWCAP2) & HWCAP2_SME) != 0;
}

/* 只做环境展示。本交付不依赖 SME2。 */
int sme_ai_has_sme2(void) {
    return (getauxval(AT_HWCAP2) & HWCAP2_SME2) != 0;
}

/*
 * 判断当前 SME1 后端是否满足本库要求。
 *
 * 注意：支持 SME 不等于一定能跑当前微内核。
 * 当前汇编微内核写死 16x16 tile，所以还要求 cntw == 16。
 */
static int sme1_backend_ready(void) {
    return sme_ai_has_sme() && sme_ai_cntw() == SME1_TILE;
}

/* 把 FP32 中间结果转成 FP16，供下一层 FP16 Linear 继续使用。 */
static void convert_fp32_to_fp16(const float *src, __fp16 *dst, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        dst[i] = (__fp16) src[i];
    }
}

/* 检查工作区是否存在、维度是否合法、是否超过初始化时设置的最大容量。 */
static sme_ai_status check_workspace(const sme_ai_workspace *ws, uint64_t batch, uint64_t in_features, uint64_t out_features) {
    if (ws == NULL || batch == 0 || in_features == 0 || out_features == 0) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (batch > ws->max_batch || in_features > ws->max_in_features || out_features > ws->max_out_features) {
        return SME_AI_WORKSPACE_TOO_SMALL;
    }
    return SME_AI_OK;
}

/*
 * 使用 SVE 复制 FP32 数组。
 * svwhilelt 生成“还没越界”的谓词，因此即使 count 不是向量长度整数倍也安全。
 */
static void copy_f32(float *dst, const float *src, uint64_t count) {
    uint64_t i = 0;
    while (i < count) {
        svbool_t pg = svwhilelt_b32_u64(i, count);
        svfloat32_t v = svld1_f32(pg, src + i);
        svst1_f32(pg, dst + i, v);
        i += svcntw();
    }
}

/* 对每一行加同一个 bias 向量：output[row][col] += bias[col]。 */
static void add_bias_f32(float *output, const float *bias, uint64_t rows, uint64_t cols) {
    if (output == NULL || bias == NULL) {
        return;
    }
    for (uint64_t i = 0; i < rows; ++i) {
        uint64_t j = 0;
        while (j < cols) {
            svbool_t pg = svwhilelt_b32_u64(j, cols);
            svfloat32_t outv = svld1_f32(pg, output + i * cols + j);
            svfloat32_t biasv = svld1_f32(pg, bias + j);
            svst1_f32(pg, output + i * cols + j, svadd_f32_x(pg, outv, biasv));
            j += svcntw();
        }
    }
}

/* 根据 act 枚举选择对应激活函数，避免 Linear+激活写多份重复代码。 */
static void apply_activation_inplace(float *data, uint64_t count, sme_ai_activation_kind act) {
    if (data == NULL) {
        return;
    }
    switch (act) {
        case SME_AI_ACT_NONE:
            return;
        case SME_AI_ACT_RELU:
            sme_ai_relu_inplace_f32(data, count);
            return;
        case SME_AI_ACT_GELU:
            sme_ai_gelu_inplace_f32(data, count);
            return;
        case SME_AI_ACT_SILU:
            sme_ai_silu_inplace_f32(data, count);
            return;
        default:
            return;
    }
}

/*
 * 最朴素的参考矩阵乘：
 * output[row, col] = sum_k input[row, k] * weight[k, col]
 *
 * 这里不用 SME，目的是作为正确性对照。
 */
static void scalar_matmul_f16f16_f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    float *output) {
    for (uint64_t row = 0; row < batch; ++row) {
        for (uint64_t col = 0; col < out_features; ++col) {
            float acc = 0.0f;
            for (uint64_t k = 0; k < in_features; ++k) {
                acc += (float) input[row * in_features + k] * (float) weight[k * out_features + col];
            }
            output[row * out_features + col] = acc;
        }
    }
}

/*
 * 将原始 FP16 权重预打包为 SME 微内核友好的布局。
 *
 * 原始 weight 布局：
 *   weight[k][n]，也就是 [in_features, out_features]
 *
 * 打包后按 16 个输出列为一个 tile：
 *   packed_weight[n_tile][k_pad][16]
 *
 * 如果 in_features 或 out_features 不是 16 的整数倍，用 0 补齐。
 * 补 0 不会改变数学结果，因为多出来的乘法贡献为 0。
 */
static void prepack_weight_fp16_to_sme_tiles(
    const __fp16 *weight,
    float *packed_weight,
    uint64_t in_features,
    uint64_t out_features) {
    uint64_t kpad = round_up_u64(in_features, SME1_TILE);
    uint64_t npad = round_up_u64(out_features, SME1_TILE);
    uint64_t tile_stride = kpad * SME1_TILE;

    memset(packed_weight, 0, kpad * npad * sizeof(float));
    for (uint64_t nb = 0; nb < npad; nb += SME1_TILE) {
        uint64_t nblk = (nb + SME1_TILE <= out_features) ? SME1_TILE : (out_features - nb);
        float *tile = packed_weight + (nb / SME1_TILE) * tile_stride;
        for (uint64_t kk = 0; kk < in_features; ++kk) {
            float *dst = tile + kk * SME1_TILE;
            for (uint64_t j = 0; j < nblk; ++j) {
                dst[j] = (float) weight[kk * out_features + nb + j];
            }
        }
    }
}

/*
 * 打包 A 矩阵的一个 16 行 tile。
 *
 * A 是 input，原始布局为 [batch, k]。
 * SME 微内核希望读取 [k_pad, 16]，也就是每个 k 对应 16 个 batch 行。
 *
 * mb 表示当前处理从第几行开始。
 * mblk 是当前 tile 实际有几行；最后一个 tile 可能不足 16 行，不足部分补 0。
 */
static void pack_a_tile_fp16_to_f32(
    const __fp16 *a,
    float *a_pack,
    uint64_t k,
    uint64_t mb,
    uint64_t mblk,
    uint64_t kpad) {
    memset(a_pack, 0, kpad * SME1_TILE * sizeof(float));
    for (uint64_t kk = 0; kk < k; ++kk) {
        const __fp16 *src = a + mb * k + kk;
        float *dst = a_pack + kk * SME1_TILE;
        for (uint64_t i = 0; i < mblk; ++i) {
            dst[i] = (float) src[i * k];
        }
    }
}

/* 根据输出列 tile 编号 nb，找到对应的 packed B 权重 tile。 */
static const float *packed_b_tile_ptr(const float *packed_weight, uint64_t kpad, uint64_t nb) {
    return packed_weight + (nb / SME1_TILE) * (kpad * SME1_TILE);
}

/* 进入 SME streaming mode。ZA 矩阵累加器需要在该模式下使用。 */
static inline void sme_streaming_start(void) {
    __asm__ volatile("smstart");
}

/* 退出 SME streaming mode。每段 SME 计算结束后都应该退出。 */
static inline void sme_streaming_stop(void) {
    __asm__ volatile("smstop");
}

/*
 * SME1 GEMM 主实现。
 *
 * 计算：
 *   C[m, n] = A[m, k] x B[k, n]
 *
 * 输入：
 * - a: 原始 FP16 A 矩阵，布局 [m, k]。
 * - b_packed: 已预打包的 B 矩阵，布局适合 16x16 SME 微内核。
 * - c: FP32 输出矩阵，布局 [m, n]。
 *
 * 处理方式：
 * 1. 每次取 A 的 16 行，打包成 [k_pad, 16]。
 * 2. 每次取 B 的 16 列 packed tile。
 * 3. 调用汇编函数 sme1_gemm_tile_16x16_*，内部用 FMOPA 做外积累加。
 * 4. 如果最后一块不足 16 行或 16 列，则先写到 gemm_c_tile，再拷贝有效部分。
 */
static sme_ai_status gemm_sme1_fp16_packed_b(
    sme_ai_workspace *ws,
    const __fp16 *a,
    const float *b_packed,
    uint64_t packed_k_pad,
    uint64_t packed_n_pad,
    float *c,
    uint64_t m,
    uint64_t k,
    uint64_t n) {
    uint64_t npad;
    uint64_t kpad;

    if (ws == NULL || a == NULL || b_packed == NULL || c == NULL || m == 0 || k == 0 || n == 0) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (!sme1_backend_ready()) {
        return SME_AI_UNSUPPORTED;
    }

    npad = round_up_u64(n, SME1_TILE);
    kpad = round_up_u64(k, SME1_TILE);
    if (kpad > ws->max_k_pad || npad > ws->max_n_pad) {
        return SME_AI_WORKSPACE_TOO_SMALL;
    }
    if (packed_k_pad < kpad || packed_n_pad < npad) {
        return SME_AI_BAD_ARGUMENT;
    }

    for (uint64_t mb = 0; mb < m; mb += SME1_TILE) {
        uint64_t mblk = (mb + SME1_TILE <= m) ? SME1_TILE : (m - mb);
        pack_a_tile_fp16_to_f32(a, ws->a_mod, k, mb, mblk, kpad);

        /*
         * 快路径：
         * - 当前 A tile 正好 16 行；
         * - n 已经是 16 的整数倍；
         * 这样 SME 微内核可以直接把 ZA 结果写入最终 C 矩阵，不需要临时 tile。
         */
        if (mblk == SME1_TILE && n == npad) {
            uint64_t row_stride_bytes = n * sizeof(float);
            sme_streaming_start();
            for (uint64_t nb = 0; nb < n; nb += SME1_TILE) {
                const float *b_tile = packed_b_tile_ptr(b_packed, packed_k_pad, nb);
                sme1_gemm_tile_16x16_strided_k(ws->a_mod, b_tile, c + mb * n + nb, kpad, row_stride_bytes);
            }
            sme_streaming_stop();
        } else {
            /*
             * 慢路径/边界路径：
             * 最后一块可能不足 16 行或 16 列。
             * SME 微内核仍然计算完整 16x16，然后只拷贝真实有效区域。
             */
            for (uint64_t nb = 0; nb < n; nb += SME1_TILE) {
                uint64_t nblk = (nb + SME1_TILE <= n) ? SME1_TILE : (n - nb);
                const float *b_tile = packed_b_tile_ptr(b_packed, packed_k_pad, nb);

                sme_streaming_start();
                sme1_gemm_tile_16x16_k(ws->a_mod, b_tile, ws->gemm_c_tile, kpad);
                sme_streaming_stop();

                for (uint64_t i = 0; i < mblk; ++i) {
                    for (uint64_t j = 0; j < nblk; ++j) {
                        c[(mb + i) * n + (nb + j)] = ws->gemm_c_tile[i * SME1_TILE + j];
                    }
                }
            }
        }
    }
    return SME_AI_OK;
}

/* Linear 的 SME packed 公共入口：先检查参数，再调用 GEMM，最后加 bias。 */
static sme_ai_status linear_core_packed(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const float *packed_weight,
    uint64_t packed_k_pad,
    uint64_t out_features,
    uint64_t packed_n_pad,
    const float *bias,
    float *output) {
    sme_ai_status status = check_workspace(ws, batch, in_features, out_features);
    if (status != SME_AI_OK) {
        return status;
    }
    if (input == NULL || packed_weight == NULL || output == NULL) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (!sme1_backend_ready()) {
        return SME_AI_UNSUPPORTED;
    }

    status = gemm_sme1_fp16_packed_b(ws, input, packed_weight, packed_k_pad, packed_n_pad, output, batch, in_features, out_features);
    if (status != SME_AI_OK) {
        return status;
    }
    add_bias_f32(output, bias, batch, out_features);
    return SME_AI_OK;
}

/* 参考 Linear + 激活函数的公共实现。 */
static sme_ai_status linear_activation_ref(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output,
    sme_ai_activation_kind act) {
    sme_ai_status status = sme_ai_linear_ref_f16f32(input, batch, in_features, weight, out_features, bias, output);
    if (status != SME_AI_OK) {
        return status;
    }
    apply_activation_inplace(output, batch * out_features, act);
    return SME_AI_OK;
}

/* SME packed Linear + 激活函数的公共实现。 */
static sme_ai_status linear_activation_sme_packed(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output,
    sme_ai_activation_kind act) {
    sme_ai_status status = sme_ai_linear_sme_packed_f16f32(ws, input, batch, packed_weight, bias, output);
    if (status != SME_AI_OK) {
        return status;
    }
    apply_activation_inplace(output, batch * packed_weight->out_features, act);
    return SME_AI_OK;
}

/*
 * 参考两层 MLP 的公共实现：
 *   tmp = activation(input x fc1 + fc1_bias)
 *   output = tmp x fc2 + fc2_bias
 */
static sme_ai_status mlp_activation_ref(
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
    float *output,
    sme_ai_activation_kind act) {
    if (ws == NULL || input == NULL || fc1_weight == NULL || fc2_weight == NULL || output == NULL) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (batch > ws->max_batch || hidden_features > ws->max_out_features || out_features > ws->max_out_features) {
        return SME_AI_WORKSPACE_TOO_SMALL;
    }
    if (sme_ai_linear_ref_f16f32(input, batch, in_features, fc1_weight, hidden_features, fc1_bias, ws->tmp_f32) != SME_AI_OK) {
        return SME_AI_BAD_ARGUMENT;
    }
    apply_activation_inplace(ws->tmp_f32, batch * hidden_features, act);
    convert_fp32_to_fp16(ws->tmp_f32, ws->tmp_f16, batch * hidden_features);
    return sme_ai_linear_ref_f16f32(ws->tmp_f16, batch, hidden_features, fc2_weight, out_features, fc2_bias, output);
}

/* SME packed 两层 MLP 的公共实现，两个 Linear 都使用 SME packed 路径。 */
static sme_ai_status mlp_activation_sme_packed(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *fc1_weight,
    const float *fc1_bias,
    const sme_ai_packed_linear_weight *fc2_weight,
    const float *fc2_bias,
    float *output,
    sme_ai_activation_kind act) {
    sme_ai_status status;
    if (ws == NULL || input == NULL || fc1_weight == NULL || fc2_weight == NULL || output == NULL) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (batch > ws->max_batch || fc1_weight->out_features > ws->max_out_features || fc2_weight->out_features > ws->max_out_features) {
        return SME_AI_WORKSPACE_TOO_SMALL;
    }

    status = sme_ai_linear_sme_packed_f16f32(ws, input, batch, fc1_weight, fc1_bias, ws->tmp_f32);
    if (status != SME_AI_OK) {
        return status;
    }
    apply_activation_inplace(ws->tmp_f32, batch * fc1_weight->out_features, act);
    convert_fp32_to_fp16(ws->tmp_f32, ws->tmp_f16, batch * fc1_weight->out_features);
    return sme_ai_linear_sme_packed_f16f32(ws, ws->tmp_f16, batch, fc2_weight, fc2_bias, output);
}

sme_ai_status sme_ai_workspace_init(sme_ai_workspace *ws, uint64_t max_batch, uint64_t max_in_features, uint64_t max_out_features) {
    uint64_t packed_weight_elems;
    if (ws == NULL || max_batch == 0 || max_in_features == 0 || max_out_features == 0) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (!sme_ai_has_sme()) {
        return SME_AI_UNSUPPORTED;
    }

    memset(ws, 0, sizeof(*ws));
    ws->cntb = sme_ai_cntb();
    ws->cntw = sme_ai_cntw();
    ws->max_batch = max_batch;
    ws->max_in_features = max_in_features;
    ws->max_out_features = max_out_features;
    ws->max_h = max_in_features;
    ws->max_m_pad = round_up_u64(max_batch, SME1_TILE);
    ws->max_k_pad = round_up_u64(max_in_features, SME1_TILE);
    ws->max_n_pad = round_up_u64(max_out_features, SME1_TILE);

    /*
     * 下面一次性申请所有常用中间缓冲区。
     * 这样后续 benchmark 调用算子时，不会反复申请/释放内存。
     */
    packed_weight_elems = ws->max_k_pad * ws->max_n_pad;
    ws->a_pairs = aligned_calloc_bytes(64, max_batch * max_in_features * sizeof(float));
    ws->a_mod = aligned_calloc_bytes(64, ws->max_k_pad * SME1_TILE * sizeof(float));
    ws->b_pack = aligned_calloc_bytes(64, packed_weight_elems * sizeof(float));
    ws->tmp_f32 = aligned_calloc_bytes(64, max_batch * max_out_features * sizeof(float));
    ws->tmp2_f32 = aligned_calloc_bytes(64, max_batch * max_out_features * sizeof(float));
    ws->tmp_f16 = aligned_calloc_bytes(64, max_batch * max_out_features * sizeof(__fp16));
    ws->gemm_c_tile = aligned_calloc_bytes(64, SME1_TILE * SME1_TILE * sizeof(float));
    ws->attn_scores = aligned_calloc_bytes(64, max_batch * max_batch * sizeof(float));

    if (ws->a_pairs == NULL || ws->a_mod == NULL || ws->b_pack == NULL || ws->tmp_f32 == NULL ||
        ws->tmp2_f32 == NULL || ws->tmp_f16 == NULL || ws->gemm_c_tile == NULL || ws->attn_scores == NULL) {
        sme_ai_workspace_destroy(ws);
        return SME_AI_ALLOC_FAILED;
    }
    return SME_AI_OK;
}

/* 释放工作区中所有缓冲区，并把结构体清零，防止悬空指针被误用。 */
void sme_ai_workspace_destroy(sme_ai_workspace *ws) {
    if (ws == NULL) {
        return;
    }
    free(ws->a_pairs);
    free(ws->a_mod);
    free(ws->b_pack);
    free(ws->tmp_f32);
    free(ws->tmp2_f32);
    free(ws->tmp_f16);
    free(ws->gemm_c_tile);
    free(ws->attn_scores);
    memset(ws, 0, sizeof(*ws));
}

sme_ai_status sme_ai_packed_linear_weight_init(
    sme_ai_packed_linear_weight *packed,
    const __fp16 *weight,
    uint64_t in_features,
    uint64_t out_features) {
    if (packed == NULL || weight == NULL || in_features == 0 || out_features == 0) {
        return SME_AI_BAD_ARGUMENT;
    }

    /*
     * k_pad/n_pad 是为了适配 16x16 SME tile。
     * 例如 out_features = 30 时，n_pad = 32，最后 2 列补 0。
     */
    memset(packed, 0, sizeof(*packed));
    packed->k_pad = round_up_u64(in_features, SME1_TILE);
    packed->n_pad = round_up_u64(out_features, SME1_TILE);
    packed->packed_weight = aligned_calloc_bytes(64, packed->k_pad * packed->n_pad * sizeof(float));
    if (packed->packed_weight == NULL) {
        return SME_AI_ALLOC_FAILED;
    }

    packed->in_features = in_features;
    packed->out_features = out_features;
    prepack_weight_fp16_to_sme_tiles(weight, packed->packed_weight, in_features, out_features);
    return SME_AI_OK;
}

/* 释放 packed_weight 内存，并清空元数据。 */
void sme_ai_packed_linear_weight_destroy(sme_ai_packed_linear_weight *packed) {
    if (packed == NULL) {
        return;
    }
    free(packed->packed_weight);
    memset(packed, 0, sizeof(*packed));
}

sme_ai_status sme_ai_linear_ref_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output) {
    if (input == NULL || weight == NULL || output == NULL || batch == 0 || in_features == 0 || out_features == 0) {
        return SME_AI_BAD_ARGUMENT;
    }
    /*
     * ref 路径就是普通三重循环矩阵乘。
     * 结果用于和 SME 路径比较，确认优化后没有算错。
     */
    scalar_matmul_f16f16_f32(input, batch, in_features, weight, out_features, output);
    add_bias_f32(output, bias, batch, out_features);
    return SME_AI_OK;
}

sme_ai_status sme_ai_linear_sme_simple_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output) {
    sme_ai_status status = check_workspace(ws, batch, in_features, out_features);
    if (status != SME_AI_OK) {
        return status;
    }
    if (weight == NULL) {
        return SME_AI_BAD_ARGUMENT;
    }
    /*
     * simple 版本每次调用都重新打包 weight。
     * 优点是调用方便；缺点是如果同一权重反复使用，会浪费打包时间。
     */
    prepack_weight_fp16_to_sme_tiles(weight, ws->b_pack, in_features, out_features);
    return linear_core_packed(
        ws,
        input,
        batch,
        in_features,
        ws->b_pack,
        round_up_u64(in_features, SME1_TILE),
        out_features,
        round_up_u64(out_features, SME1_TILE),
        bias,
        output);
}

sme_ai_status sme_ai_linear_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output) {
    if (packed_weight == NULL || packed_weight->packed_weight == NULL) {
        return SME_AI_BAD_ARGUMENT;
    }
    /* packed 版本直接使用已经整理好的权重，是推理时更推荐的路径。 */
    return linear_core_packed(
        ws,
        input,
        batch,
        packed_weight->in_features,
        packed_weight->packed_weight,
        packed_weight->k_pad,
        packed_weight->out_features,
        packed_weight->n_pad,
        bias,
        output);
}

sme_ai_status sme_ai_linear_relu_ref_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output) {
    return linear_activation_ref(input, batch, in_features, weight, out_features, bias, output, SME_AI_ACT_RELU);
}

sme_ai_status sme_ai_linear_relu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output) {
    return linear_activation_sme_packed(ws, input, batch, packed_weight, bias, output, SME_AI_ACT_RELU);
}

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
    float *output) {
    return mlp_activation_ref(
        ws, input, batch, in_features, fc1_weight, hidden_features, fc1_bias, fc2_weight, out_features, fc2_bias, output, SME_AI_ACT_RELU);
}

sme_ai_status sme_ai_mlp_relu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *fc1_weight,
    const float *fc1_bias,
    const sme_ai_packed_linear_weight *fc2_weight,
    const float *fc2_bias,
    float *output) {
    return mlp_activation_sme_packed(ws, input, batch, fc1_weight, fc1_bias, fc2_weight, fc2_bias, output, SME_AI_ACT_RELU);
}

void sme_ai_add_f32(float *output, const float *lhs, const float *rhs, uint64_t count) {
    uint64_t i = 0;
    if (output == NULL || lhs == NULL || rhs == NULL) {
        return;
    }
    /*
     * SVE 写法：
     * - svwhilelt_b32_u64 负责处理最后不足一个向量的尾巴。
     * - svld1_f32 读取一段 float。
     * - svadd_f32_x 做逐元素加法。
     * - svst1_f32 写回结果。
     */
    while (i < count) {
        svbool_t pg = svwhilelt_b32_u64(i, count);
        svfloat32_t a = svld1_f32(pg, lhs + i);
        svfloat32_t b = svld1_f32(pg, rhs + i);
        svst1_f32(pg, output + i, svadd_f32_x(pg, a, b));
        i += svcntw();
    }
}

void sme_ai_mul_f32(float *output, const float *lhs, const float *rhs, uint64_t count) {
    uint64_t i = 0;
    if (output == NULL || lhs == NULL || rhs == NULL) {
        return;
    }
    /* 与 add 类似，只是把加法换成乘法。 */
    while (i < count) {
        svbool_t pg = svwhilelt_b32_u64(i, count);
        svfloat32_t a = svld1_f32(pg, lhs + i);
        svfloat32_t b = svld1_f32(pg, rhs + i);
        svst1_f32(pg, output + i, svmul_f32_x(pg, a, b));
        i += svcntw();
    }
}

void sme_ai_residual_add_f32(float *output, const float *input, const float *residual, uint64_t count) {
    sme_ai_add_f32(output, input, residual, count);
}

void sme_ai_add_bias_rowwise_f32(float *output, const float *input, const float *bias, uint64_t rows, uint64_t cols) {
    if (output == NULL || input == NULL) {
        return;
    }
    /*
     * 支持 output 和 input 指向同一块内存。
     * 如果不是同一块，先复制 input，再在 output 上加 bias。
     */
    if (output != input) {
        copy_f32(output, input, rows * cols);
    }
    add_bias_f32(output, bias, rows, cols);
}

void sme_ai_relu_inplace_f32(float *data, uint64_t count) {
    uint64_t i = 0;
    if (data == NULL) {
        return;
    }
    /* ReLU 可以很好地向量化：每个元素和 0 取最大值。 */
    while (i < count) {
        svbool_t pg = svwhilelt_b32_u64(i, count);
        svfloat32_t x = svld1_f32(pg, data + i);
        svst1_f32(pg, data + i, svmax_n_f32_m(pg, x, 0.0f));
        i += svcntw();
    }
}

void sme_ai_sigmoid_inplace_f32(float *data, uint64_t count) {
    if (data == NULL) {
        return;
    }
    /*
     * sigmoid/tanh/GELU/SiLU 涉及 exp/tanh 等数学函数。
     * 这里为了保证实现简单清晰，使用标量 math 库逐个计算。
     */
    for (uint64_t i = 0; i < count; ++i) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}

void sme_ai_tanh_inplace_f32(float *data, uint64_t count) {
    if (data == NULL) {
        return;
    }
    for (uint64_t i = 0; i < count; ++i) {
        data[i] = tanhf(data[i]);
    }
}

void sme_ai_gelu_inplace_f32(float *data, uint64_t count) {
    if (data == NULL) {
        return;
    }
    for (uint64_t i = 0; i < count; ++i) {
        data[i] = gelu_approx(data[i]);
    }
}

void sme_ai_silu_inplace_f32(float *data, uint64_t count) {
    if (data == NULL) {
        return;
    }
    for (uint64_t i = 0; i < count; ++i) {
        float x = data[i];
        data[i] = x / (1.0f + expf(-x));
    }
}

void sme_ai_softmax_rowwise_f32(float *output, const float *input, uint64_t rows, uint64_t cols) {
    sme_ai_softmax_masked_rowwise_f32(output, input, NULL, rows, cols);
}

void sme_ai_softmax_masked_rowwise_f32(float *output, const float *input, const float *mask, uint64_t rows, uint64_t cols) {
    if (output == NULL || input == NULL || rows == 0 || cols == 0) {
        return;
    }
    for (uint64_t i = 0; i < rows; ++i) {
        float max_value = -INFINITY;
        float sum = 0.0f;
        /*
         * softmax 为了数值稳定，先减去这一行的最大值：
         * softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
         * 这样可以避免 exp 输入过大导致溢出。
         */
        for (uint64_t j = 0; j < cols; ++j) {
            float value = input[i * cols + j];
            if (mask != NULL) {
                value += mask[i * cols + j];
            }
            if (value > max_value) {
                max_value = value;
            }
        }
        for (uint64_t j = 0; j < cols; ++j) {
            float value = input[i * cols + j];
            if (mask != NULL) {
                value += mask[i * cols + j];
            }
            {
                float ex = expf(value - max_value);
                output[i * cols + j] = ex;
                sum += ex;
            }
        }
        if (sum == 0.0f) {
            /* 理论上很少发生；保留保护逻辑，避免除以 0。 */
            for (uint64_t j = 0; j < cols; ++j) {
                output[i * cols + j] = 0.0f;
            }
        } else {
            /* 除以 sum 的归一化步骤用 SVE 向量化。 */
            uint64_t j = 0;
            while (j < cols) {
                svbool_t pg = svwhilelt_b32_u64(j, cols);
                svfloat32_t outv = svld1_f32(pg, output + i * cols + j);
                svst1_f32(pg, output + i * cols + j, svdiv_n_f32_x(pg, outv, sum));
                j += svcntw();
            }
        }
    }
}

void sme_ai_softmax_causal_rowwise_f32(float *output, const float *input, uint64_t rows, uint64_t cols) {
    if (output == NULL || input == NULL || rows == 0 || cols == 0) {
        return;
    }
    for (uint64_t i = 0; i < rows; ++i) {
        float max_value = -INFINITY;
        float sum = 0.0f;
        /*
         * causal mask 用于自回归模型。
         * 第 i 行只能看 j <= i 的位置；j > i 的“未来 token”被设成极小值。
         */
        for (uint64_t j = 0; j < cols; ++j) {
            float value = (j > i) ? -1.0e30f : input[i * cols + j];
            if (value > max_value) {
                max_value = value;
            }
        }
        for (uint64_t j = 0; j < cols; ++j) {
            float value = (j > i) ? -1.0e30f : input[i * cols + j];
            float ex = expf(value - max_value);
            output[i * cols + j] = ex;
            sum += ex;
        }
        if (sum == 0.0f) {
            for (uint64_t j = 0; j < cols; ++j) {
                output[i * cols + j] = 0.0f;
            }
        } else {
            uint64_t j = 0;
            while (j < cols) {
                svbool_t pg = svwhilelt_b32_u64(j, cols);
                svfloat32_t outv = svld1_f32(pg, output + i * cols + j);
                svst1_f32(pg, output + i * cols + j, svdiv_n_f32_x(pg, outv, sum));
                j += svcntw();
            }
        }
    }
}

/* 用 SVE 对一行 float 求和，LayerNorm 计算均值时使用。 */
static float row_sum_f32_sve(const float *data, uint64_t cols) {
    float sum = 0.0f;
    uint64_t j = 0;
    while (j < cols) {
        svbool_t pg = svwhilelt_b32_u64(j, cols);
        svfloat32_t v = svld1_f32(pg, data + j);
        sum += svaddv_f32(pg, v);
        j += svcntw();
    }
    return sum;
}

/* 用 SVE 对一行 float 的平方求和，RMSNorm/LayerNorm 计算方差时使用。 */
static float row_sumsq_f32_sve(const float *data, uint64_t cols) {
    float sum = 0.0f;
    uint64_t j = 0;
    while (j < cols) {
        svbool_t pg = svwhilelt_b32_u64(j, cols);
        svfloat32_t v = svld1_f32(pg, data + j);
        sum += svaddv_f32(pg, svmul_f32_x(pg, v, v));
        j += svcntw();
    }
    return sum;
}

void sme_ai_layernorm_f32(
    float *output,
    const float *input,
    const float *gamma,
    const float *beta,
    uint64_t rows,
    uint64_t cols,
    float eps) {
    if (output == NULL || input == NULL || rows == 0 || cols == 0) {
        return;
    }
    for (uint64_t i = 0; i < rows; ++i) {
        const float *in_row = input + i * cols;
        float *out_row = output + i * cols;
        /*
         * LayerNorm 对每一行独立归一化：
         * mean = average(x)
         * var  = average((x - mean)^2)
         * y    = (x - mean) / sqrt(var + eps)
         * 如果提供 gamma/beta，再执行 y = y * gamma + beta。
         */
        float mean = row_sum_f32_sve(in_row, cols) / (float) cols;
        float var = 0.0f;
        uint64_t j = 0;
        while (j < cols) {
            svbool_t pg = svwhilelt_b32_u64(j, cols);
            svfloat32_t x = svld1_f32(pg, in_row + j);
            svfloat32_t d = svsub_f32_x(pg, x, svdup_n_f32(mean));
            var += svaddv_f32(pg, svmul_f32_x(pg, d, d));
            j += svcntw();
        }
        var /= (float) cols;
        {
            float inv_std = 1.0f / sqrtf(var + eps);
            j = 0;
            while (j < cols) {
                svbool_t pg = svwhilelt_b32_u64(j, cols);
                svfloat32_t x = svld1_f32(pg, in_row + j);
                svfloat32_t y = svmul_n_f32_x(pg, svsub_f32_x(pg, x, svdup_n_f32(mean)), inv_std);
                if (gamma != NULL) {
                    y = svmul_f32_x(pg, y, svld1_f32(pg, gamma + j));
                }
                if (beta != NULL) {
                    y = svadd_f32_x(pg, y, svld1_f32(pg, beta + j));
                }
                svst1_f32(pg, out_row + j, y);
                j += svcntw();
            }
        }
    }
}

void sme_ai_rmsnorm_f32(
    float *output,
    const float *input,
    const float *weight,
    uint64_t rows,
    uint64_t cols,
    float eps) {
    if (output == NULL || input == NULL || rows == 0 || cols == 0) {
        return;
    }
    for (uint64_t i = 0; i < rows; ++i) {
        const float *in_row = input + i * cols;
        float *out_row = output + i * cols;
        /*
         * RMSNorm 不减均值，只除以均方根：
         * rms = sqrt(average(x^2) + eps)
         * y = x / rms
         * LLaMA 系列模型常用 RMSNorm。
         */
        float mean_sq = row_sumsq_f32_sve(in_row, cols) / (float) cols;
        float inv_rms = 1.0f / sqrtf(mean_sq + eps);
        uint64_t j = 0;
        while (j < cols) {
            svbool_t pg = svwhilelt_b32_u64(j, cols);
            svfloat32_t y = svmul_n_f32_x(pg, svld1_f32(pg, in_row + j), inv_rms);
            if (weight != NULL) {
                y = svmul_f32_x(pg, y, svld1_f32(pg, weight + j));
            }
            svst1_f32(pg, out_row + j, y);
            j += svcntw();
        }
    }
}

sme_ai_status sme_ai_embedding_lookup_f16f32(
    float *output,
    const __fp16 *table,
    const uint32_t *token_ids,
    uint64_t num_tokens,
    uint64_t vocab_size,
    uint64_t hidden_size) {
    if (output == NULL || table == NULL || token_ids == NULL || num_tokens == 0 || vocab_size == 0 || hidden_size == 0) {
        return SME_AI_BAD_ARGUMENT;
    }
    /*
     * embedding lookup 本质是查表：
     * token id = 5，就取 table 的第 5 行。
     * 如果 id 越界，这里输出全 0，避免非法访问。
     */
    for (uint64_t t = 0; t < num_tokens; ++t) {
        uint32_t id = token_ids[t];
        for (uint64_t h = 0; h < hidden_size; ++h) {
            output[t * hidden_size + h] = (id < vocab_size) ? (float) table[(uint64_t) id * hidden_size + h] : 0.0f;
        }
    }
    return SME_AI_OK;
}

sme_ai_status sme_ai_linear_gelu_ref_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output) {
    return linear_activation_ref(input, batch, in_features, weight, out_features, bias, output, SME_AI_ACT_GELU);
}

sme_ai_status sme_ai_linear_gelu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output) {
    return linear_activation_sme_packed(ws, input, batch, packed_weight, bias, output, SME_AI_ACT_GELU);
}

sme_ai_status sme_ai_linear_silu_ref_f16f32(
    const __fp16 *input,
    uint64_t batch,
    uint64_t in_features,
    const __fp16 *weight,
    uint64_t out_features,
    const float *bias,
    float *output) {
    return linear_activation_ref(input, batch, in_features, weight, out_features, bias, output, SME_AI_ACT_SILU);
}

sme_ai_status sme_ai_linear_silu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *packed_weight,
    const float *bias,
    float *output) {
    return linear_activation_sme_packed(ws, input, batch, packed_weight, bias, output, SME_AI_ACT_SILU);
}

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
    float *output) {
    return mlp_activation_ref(
        ws, input, batch, in_features, fc1_weight, hidden_features, fc1_bias, fc2_weight, out_features, fc2_bias, output, SME_AI_ACT_GELU);
}

sme_ai_status sme_ai_mlp_gelu_sme_packed_f16f32(
    sme_ai_workspace *ws,
    const __fp16 *input,
    uint64_t batch,
    const sme_ai_packed_linear_weight *fc1_weight,
    const float *fc1_bias,
    const sme_ai_packed_linear_weight *fc2_weight,
    const float *fc2_bias,
    float *output) {
    return mlp_activation_sme_packed(ws, input, batch, fc1_weight, fc1_bias, fc2_weight, fc2_bias, output, SME_AI_ACT_GELU);
}

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
    float *output) {
    if (ws == NULL || input == NULL || gate_weight == NULL || up_weight == NULL || down_weight == NULL || output == NULL) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (batch > ws->max_batch || hidden_features > ws->max_out_features || out_features > ws->max_out_features) {
        return SME_AI_WORKSPACE_TOO_SMALL;
    }

    if (sme_ai_linear_ref_f16f32(input, batch, in_features, gate_weight, hidden_features, gate_bias, ws->tmp_f32) != SME_AI_OK ||
        sme_ai_linear_ref_f16f32(input, batch, in_features, up_weight, hidden_features, up_bias, ws->tmp2_f32) != SME_AI_OK) {
        return SME_AI_BAD_ARGUMENT;
    }

    /*
     * SwiGLU 结构：
     * gate = SiLU(input x gate_weight)
     * up   = input x up_weight
     * hidden = gate * up
     * output = hidden x down_weight
     */
    sme_ai_silu_inplace_f32(ws->tmp_f32, batch * hidden_features);
    sme_ai_mul_f32(ws->tmp_f32, ws->tmp_f32, ws->tmp2_f32, batch * hidden_features);
    convert_fp32_to_fp16(ws->tmp_f32, ws->tmp_f16, batch * hidden_features);
    return sme_ai_linear_ref_f16f32(ws->tmp_f16, batch, hidden_features, down_weight, out_features, down_bias, output);
}

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
    float *output) {
    sme_ai_status status;
    if (ws == NULL || input == NULL || gate_weight == NULL || up_weight == NULL || down_weight == NULL || output == NULL) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (batch > ws->max_batch ||
        gate_weight->out_features > ws->max_out_features ||
        up_weight->out_features > ws->max_out_features ||
        down_weight->out_features > ws->max_out_features) {
        return SME_AI_WORKSPACE_TOO_SMALL;
    }
    if (gate_weight->out_features != up_weight->out_features || gate_weight->out_features != down_weight->in_features) {
        return SME_AI_BAD_ARGUMENT;
    }

    status = sme_ai_linear_sme_packed_f16f32(ws, input, batch, gate_weight, gate_bias, ws->tmp_f32);
    if (status == SME_AI_OK) {
        status = sme_ai_linear_sme_packed_f16f32(ws, input, batch, up_weight, up_bias, ws->tmp2_f32);
    }
    if (status != SME_AI_OK) {
        return status;
    }

    /* packed 版本和 ref 版本的数学流程相同，只是三个 Linear 走 SME packed 路径。 */
    sme_ai_silu_inplace_f32(ws->tmp_f32, batch * gate_weight->out_features);
    sme_ai_mul_f32(ws->tmp_f32, ws->tmp_f32, ws->tmp2_f32, batch * gate_weight->out_features);
    convert_fp32_to_fp16(ws->tmp_f32, ws->tmp_f16, batch * gate_weight->out_features);
    return sme_ai_linear_sme_packed_f16f32(ws, ws->tmp_f16, batch, down_weight, down_bias, output);
}

/* SVE 点积，用于 attention 中 Q 和 K 的相似度计算。 */
static float dot_f32_sve(const float *lhs, const float *rhs, uint64_t count) {
    uint64_t i = 0;
    svfloat32_t acc = svdup_n_f32(0.0f);
    while (i < count) {
        svbool_t pg = svwhilelt_b32_u64(i, count);
        svfloat32_t a = svld1_f32(pg, lhs + i);
        svfloat32_t b = svld1_f32(pg, rhs + i);
        acc = svmla_f32_m(pg, acc, a, b);
        i += svcntw();
    }
    return svaddv_f32(svptrue_b32(), acc);
}

/*
 * 根据 softmax 后的 attention score，对 V 做加权求和。
 *
 * 对每个 query 位置 i：
 * context[i] = sum_j score[i, j] * V[j]
 */
static void attention_weighted_sum_f32(
    float *context,
    const float *score_head,
    const float *v,
    uint64_t seq_len,
    uint64_t model_dim,
    uint64_t head_offset,
    uint64_t head_dim) {
    for (uint64_t i = 0; i < seq_len; ++i) {
        float *ctx_row = context + i * model_dim + head_offset;
        uint64_t d = 0;
        while (d < head_dim) {
            svbool_t pg = svwhilelt_b32_u64(d, head_dim);
            svst1_f32(pg, ctx_row + d, svdup_n_f32(0.0f));
            d += svcntw();
        }

        for (uint64_t j = 0; j < seq_len; ++j) {
            float score = score_head[i * seq_len + j];
            const float *v_row = v + j * model_dim + head_offset;
            d = 0;
            while (d < head_dim) {
                svbool_t pg = svwhilelt_b32_u64(d, head_dim);
                svfloat32_t acc = svld1_f32(pg, ctx_row + d);
                svfloat32_t vv = svld1_f32(pg, v_row + d);
                acc = svadd_f32_x(pg, acc, svmul_n_f32_x(pg, vv, score));
                svst1_f32(pg, ctx_row + d, acc);
                d += svcntw();
            }
        }
    }
}

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
    float *output) {
    uint64_t head_dim;
    float *q;
    float *k;
    float *v;
    float *scores;
    float *context;
    __fp16 *context_f16;
    sme_ai_status status = SME_AI_OK;

    if (ws == NULL || input == NULL || q_weight == NULL || k_weight == NULL || v_weight == NULL || out_weight == NULL || output == NULL ||
        seq_len == 0 || model_dim == 0 || num_heads == 0 || model_dim % num_heads != 0) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (seq_len > ws->max_batch || model_dim > ws->max_in_features || model_dim > ws->max_out_features) {
        return SME_AI_WORKSPACE_TOO_SMALL;
    }

    head_dim = model_dim / num_heads;
    q = ws->tmp_f32;
    k = ws->tmp2_f32;
    v = output;
    scores = ws->attn_scores;
    context = ws->a_pairs;
    context_f16 = ws->tmp_f16;

    /*
     * 1. 先做 Q/K/V 三个投影。
     *    ref 版本使用普通参考 Linear，便于和 SME 版本做结果对比。
     */
    status = sme_ai_linear_ref_f16f32(input, seq_len, model_dim, q_weight, model_dim, q_bias, q);
    if (status != SME_AI_OK) {
        return status;
    }
    status = sme_ai_linear_ref_f16f32(input, seq_len, model_dim, k_weight, model_dim, k_bias, k);
    if (status != SME_AI_OK) {
        return status;
    }
    status = sme_ai_linear_ref_f16f32(input, seq_len, model_dim, v_weight, model_dim, v_bias, v);
    if (status != SME_AI_OK) {
        return status;
    }

    /*
     * 2. 分 head 计算 attention。
     *
     * 对每个 head：
     * - score[i, j] = dot(Q_i, K_j) / sqrt(head_dim)
     * - 对 score 的每一行做 softmax
     * - context_i = sum_j score[i, j] * V_j
     */
    for (uint64_t h = 0; h < num_heads; ++h) {
        uint64_t head_offset = h * head_dim;
        float scale = 1.0f / sqrtf((float) head_dim);
        for (uint64_t i = 0; i < seq_len; ++i) {
            const float *q_row = q + i * model_dim + head_offset;
            for (uint64_t j = 0; j < seq_len; ++j) {
                const float *k_row = k + j * model_dim + head_offset;
                scores[i * seq_len + j] = dot_f32_sve(q_row, k_row, head_dim) * scale;
            }
        }
        if (causal) {
            sme_ai_softmax_causal_rowwise_f32(scores, scores, seq_len, seq_len);
        } else {
            sme_ai_softmax_rowwise_f32(scores, scores, seq_len, seq_len);
        }
        attention_weighted_sum_f32(context, scores, v, seq_len, model_dim, head_offset, head_dim);
    }

    /*
     * 3. 最后做输出投影。
     *    由于 Linear 输入类型是 FP16，这里把 FP32 context 转回 FP16。
     */
    convert_fp32_to_fp16(context, context_f16, seq_len * model_dim);
    return sme_ai_linear_ref_f16f32(context_f16, seq_len, model_dim, out_weight, model_dim, out_bias, output);
}

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
    float *output) {
    uint64_t head_dim;
    float *q;
    float *k;
    float *v;
    float *scores;
    float *context;
    __fp16 *context_f16;
    sme_ai_status status = SME_AI_OK;

    if (ws == NULL || input == NULL || q_weight == NULL || k_weight == NULL || v_weight == NULL || out_weight == NULL || output == NULL ||
        seq_len == 0 || model_dim == 0 || num_heads == 0 || model_dim % num_heads != 0) {
        return SME_AI_BAD_ARGUMENT;
    }
    if (seq_len > ws->max_batch || model_dim > ws->max_in_features || model_dim > ws->max_out_features) {
        return SME_AI_WORKSPACE_TOO_SMALL;
    }
    if (q_weight->in_features != model_dim || q_weight->out_features != model_dim ||
        k_weight->in_features != model_dim || k_weight->out_features != model_dim ||
        v_weight->in_features != model_dim || v_weight->out_features != model_dim ||
        out_weight->in_features != model_dim || out_weight->out_features != model_dim) {
        return SME_AI_BAD_ARGUMENT;
    }

    head_dim = model_dim / num_heads;
    q = ws->tmp_f32;
    k = ws->tmp2_f32;
    v = output;
    scores = ws->attn_scores;
    context = ws->a_pairs;
    context_f16 = ws->tmp_f16;

    /*
     * SME attention 和 ref attention 的数学流程相同。
     * 不同点在于 Q/K/V/O 四个投影都调用 sme_ai_linear_sme_packed_f16f32，
     * 也就是进入 SME1 16x16 FMOPA GEMM 微内核。
     */
    status = sme_ai_linear_sme_packed_f16f32(ws, input, seq_len, q_weight, q_bias, q);
    if (status != SME_AI_OK) {
        return status;
    }
    status = sme_ai_linear_sme_packed_f16f32(ws, input, seq_len, k_weight, k_bias, k);
    if (status != SME_AI_OK) {
        return status;
    }
    status = sme_ai_linear_sme_packed_f16f32(ws, input, seq_len, v_weight, v_bias, v);
    if (status != SME_AI_OK) {
        return status;
    }

    /* score、softmax、V 加权求和仍然使用通用 SVE/标量实现。 */
    for (uint64_t h = 0; h < num_heads; ++h) {
        uint64_t head_offset = h * head_dim;
        float scale = 1.0f / sqrtf((float) head_dim);
        for (uint64_t i = 0; i < seq_len; ++i) {
            const float *q_row = q + i * model_dim + head_offset;
            for (uint64_t j = 0; j < seq_len; ++j) {
                const float *k_row = k + j * model_dim + head_offset;
                scores[i * seq_len + j] = dot_f32_sve(q_row, k_row, head_dim) * scale;
            }
        }
        if (causal) {
            sme_ai_softmax_causal_rowwise_f32(scores, scores, seq_len, seq_len);
        } else {
            sme_ai_softmax_rowwise_f32(scores, scores, seq_len, seq_len);
        }
        attention_weighted_sum_f32(context, scores, v, seq_len, model_dim, head_offset, head_dim);
    }

    convert_fp32_to_fp16(context, context_f16, seq_len * model_dim);
    return sme_ai_linear_sme_packed_f16f32(ws, context_f16, seq_len, out_weight, out_bias, output);
}

/*
 * 参考 Conv2D。
 *
 * 这是最直观的 7 层循环实现：
 * batch -> out_channel -> out_h -> out_w -> in_channel -> kernel_h -> kernel_w
 *
 * 优点：容易理解，结果可靠。
 * 缺点：速度慢，主要用于正确性对照。
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
    float *output) {
    uint64_t out_h = conv_out_dim(in_h, kernel_h, stride_h, pad_h);
    uint64_t out_w = conv_out_dim(in_w, kernel_w, stride_w, pad_w);
    if (input == NULL || weight == NULL || output == NULL || batch == 0 || in_channels == 0 || in_h == 0 || in_w == 0 ||
        out_channels == 0 || kernel_h == 0 || kernel_w == 0 || out_h == 0 || out_w == 0) {
        return SME_AI_BAD_ARGUMENT;
    }
    for (uint64_t n = 0; n < batch; ++n) {
        for (uint64_t oc = 0; oc < out_channels; ++oc) {
            for (uint64_t oh = 0; oh < out_h; ++oh) {
                for (uint64_t ow = 0; ow < out_w; ++ow) {
                    float acc = bias ? bias[oc] : 0.0f;
                    for (uint64_t ic = 0; ic < in_channels; ++ic) {
                        for (uint64_t kh = 0; kh < kernel_h; ++kh) {
                            for (uint64_t kw = 0; kw < kernel_w; ++kw) {
                                int64_t ih = (int64_t) oh * (int64_t) stride_h + (int64_t) kh - (int64_t) pad_h;
                                int64_t iw = (int64_t) ow * (int64_t) stride_w + (int64_t) kw - (int64_t) pad_w;
                                if (ih >= 0 && iw >= 0 && (uint64_t) ih < in_h && (uint64_t) iw < in_w) {
                                    uint64_t in_idx = (((n * in_channels + ic) * in_h + (uint64_t) ih) * in_w) + (uint64_t) iw;
                                    uint64_t w_idx = ((((oc * in_channels) + ic) * kernel_h + kh) * kernel_w) + kw;
                                    acc += (float) input[in_idx] * (float) weight[w_idx];
                                }
                            }
                        }
                    }
                    output[(((n * out_channels + oc) * out_h + oh) * out_w) + ow] = acc;
                }
            }
        }
    }
    return SME_AI_OK;
}

/*
 * SME Conv2D：im2col + SME Linear。
 *
 * im2col 的含义：
 * - 卷积每个输出点都会看到输入上的一个 kernel patch。
 * - 把每个 patch 拉平成一行，得到 col 矩阵：
 *   col shape = [batch * out_h * out_w, in_channels * kernel_h * kernel_w]
 * - 把卷积核也拉平成 Linear 权重：
 *   flat_weight shape = [patch_size, out_channels]
 * - 于是 Conv2D 就变成：
 *   output_matrix = col x flat_weight
 *
 * 这样可以复用已经优化好的 SME packed Linear。
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
    float *output) {
    uint64_t out_h = conv_out_dim(in_h, kernel_h, stride_h, pad_h);
    uint64_t out_w = conv_out_dim(in_w, kernel_w, stride_w, pad_w);
    uint64_t patch_size;
    uint64_t patches;
    __fp16 *col = NULL;
    __fp16 *flat_weight = NULL;
    sme_ai_packed_linear_weight packed;
    sme_ai_status status;

    if (ws == NULL || input == NULL || weight == NULL || output == NULL || batch == 0 || in_channels == 0 || in_h == 0 || in_w == 0 ||
        out_channels == 0 || kernel_h == 0 || kernel_w == 0 || out_h == 0 || out_w == 0) {
        return SME_AI_BAD_ARGUMENT;
    }

    patch_size = in_channels * kernel_h * kernel_w;
    patches = batch * out_h * out_w;
    status = check_workspace(ws, patches, patch_size, out_channels);
    if (status != SME_AI_OK) {
        return status;
    }

    /* 第一步：构造 im2col 矩阵。padding 越界位置填 0。 */
    col = aligned_calloc_bytes(64, patches * patch_size * sizeof(__fp16));
    flat_weight = aligned_calloc_bytes(64, patch_size * out_channels * sizeof(__fp16));
    memset(&packed, 0, sizeof(packed));
    if (col == NULL || flat_weight == NULL) {
        free(col);
        free(flat_weight);
        return SME_AI_ALLOC_FAILED;
    }

    for (uint64_t n = 0; n < batch; ++n) {
        for (uint64_t oh = 0; oh < out_h; ++oh) {
            for (uint64_t ow = 0; ow < out_w; ++ow) {
                uint64_t row = (n * out_h + oh) * out_w + ow;
                uint64_t col_idx = 0;
                for (uint64_t ic = 0; ic < in_channels; ++ic) {
                    for (uint64_t kh = 0; kh < kernel_h; ++kh) {
                        for (uint64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t ih = (int64_t) oh * (int64_t) stride_h + (int64_t) kh - (int64_t) pad_h;
                            int64_t iw = (int64_t) ow * (int64_t) stride_w + (int64_t) kw - (int64_t) pad_w;
                            if (ih >= 0 && iw >= 0 && (uint64_t) ih < in_h && (uint64_t) iw < in_w) {
                                uint64_t in_idx = (((n * in_channels + ic) * in_h + (uint64_t) ih) * in_w) + (uint64_t) iw;
                                col[row * patch_size + col_idx] = input[in_idx];
                            } else {
                                col[row * patch_size + col_idx] = (__fp16) 0.0f;
                            }
                            ++col_idx;
                        }
                    }
                }
            }
        }
    }

    /* 第二步：把卷积核从 [out_c, in_c, kh, kw] 改成 Linear 权重 [patch_size, out_c]。 */
    for (uint64_t ic = 0; ic < in_channels; ++ic) {
        for (uint64_t kh = 0; kh < kernel_h; ++kh) {
            for (uint64_t kw = 0; kw < kernel_w; ++kw) {
                uint64_t patch_idx = (ic * kernel_h + kh) * kernel_w + kw;
                for (uint64_t oc = 0; oc < out_channels; ++oc) {
                    uint64_t src_idx = ((((oc * in_channels) + ic) * kernel_h + kh) * kernel_w) + kw;
                    flat_weight[patch_idx * out_channels + oc] = weight[src_idx];
                }
            }
        }
    }

    /* 第三步：预打包权重，然后调用 SME packed Linear。 */
    status = sme_ai_packed_linear_weight_init(&packed, flat_weight, patch_size, out_channels);
    if (status != SME_AI_OK) {
        free(col);
        free(flat_weight);
        return status;
    }

    status = sme_ai_linear_sme_packed_f16f32(ws, col, patches, &packed, bias, ws->tmp_f32);
    if (status == SME_AI_OK) {
        /*
         * 第四步：把矩阵乘输出 [patches, out_channels]
         * 重新排回 NCHW 输出 [batch, out_channels, out_h, out_w]。
         */
        for (uint64_t n = 0; n < batch; ++n) {
            for (uint64_t oh = 0; oh < out_h; ++oh) {
                for (uint64_t ow = 0; ow < out_w; ++ow) {
                    uint64_t row = (n * out_h + oh) * out_w + ow;
                    for (uint64_t oc = 0; oc < out_channels; ++oc) {
                        output[(((n * out_channels + oc) * out_h + oh) * out_w) + ow] = ws->tmp_f32[row * out_channels + oc];
                    }
                }
            }
        }
    }

    sme_ai_packed_linear_weight_destroy(&packed);
    free(col);
    free(flat_weight);
    return status;
}

/* MaxPool：每个窗口取最大值，常用于 CNN 下采样。 */
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
    uint64_t stride_w) {
    uint64_t out_h = conv_out_dim(in_h, kernel_h, stride_h, 0);
    uint64_t out_w = conv_out_dim(in_w, kernel_w, stride_w, 0);
    if (output == NULL || input == NULL || out_h == 0 || out_w == 0) {
        return;
    }
    for (uint64_t n = 0; n < batch; ++n) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t oh = 0; oh < out_h; ++oh) {
                for (uint64_t ow = 0; ow < out_w; ++ow) {
                    float max_value = -INFINITY;
                    for (uint64_t kh = 0; kh < kernel_h; ++kh) {
                        for (uint64_t kw = 0; kw < kernel_w; ++kw) {
                            uint64_t ih = oh * stride_h + kh;
                            uint64_t iw = ow * stride_w + kw;
                            float value = input[(((n * channels + c) * in_h + ih) * in_w) + iw];
                            if (value > max_value) {
                                max_value = value;
                            }
                        }
                    }
                    output[(((n * channels + c) * out_h + oh) * out_w) + ow] = max_value;
                }
            }
        }
    }
}

/* AvgPool：每个窗口取平均值，常用于 CNN 下采样。 */
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
    uint64_t stride_w) {
    uint64_t out_h = conv_out_dim(in_h, kernel_h, stride_h, 0);
    uint64_t out_w = conv_out_dim(in_w, kernel_w, stride_w, 0);
    if (output == NULL || input == NULL || out_h == 0 || out_w == 0) {
        return;
    }
    for (uint64_t n = 0; n < batch; ++n) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t oh = 0; oh < out_h; ++oh) {
                for (uint64_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    for (uint64_t kh = 0; kh < kernel_h; ++kh) {
                        for (uint64_t kw = 0; kw < kernel_w; ++kw) {
                            uint64_t ih = oh * stride_h + kh;
                            uint64_t iw = ow * stride_w + kw;
                            sum += input[(((n * channels + c) * in_h + ih) * in_w) + iw];
                        }
                    }
                    output[(((n * channels + c) * out_h + oh) * out_w) + ow] = sum / (float) (kernel_h * kernel_w);
                }
            }
        }
    }
}
