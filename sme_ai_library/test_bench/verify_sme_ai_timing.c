#define _GNU_SOURCE

#include "sme_ai_ops.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int (*case_fn)(void);

typedef struct {
    const char *name;
    case_fn fn;
} case_entry;

typedef enum {
    LINEAR_REF = 0,
    LINEAR_SME_SIMPLE,
    LINEAR_SME_PACKED,
    LINEAR_RELU_REF,
    LINEAR_RELU_SME,
    LINEAR_GELU_REF,
    LINEAR_GELU_SME,
    LINEAR_SILU_REF,
    LINEAR_SILU_SME,
} linear_kind;

typedef enum {
    MLP_RELU_REF = 0,
    MLP_RELU_SME,
    MLP_GELU_REF,
    MLP_GELU_SME,
} mlp_kind;

typedef enum {
    SWIGLU_REF = 0,
    SWIGLU_SME,
} swiglu_kind;

typedef enum {
    ATTN_REF = 0,
    ATTN_SME,
} attention_kind;

typedef enum {
    CONV_REF = 0,
    CONV_SME,
} conv_kind;

static const char *g_case_filter = NULL;
static int g_iter_scale = 1;
static int g_no_header = 0;
static volatile float g_sink = 0.0f;

static void *aligned_calloc_bytes(size_t alignment, size_t bytes) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        return NULL;
    }
    memset(ptr, 0, bytes);
    return ptr;
}

static uint32_t next_rand_u32(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_fp16(__fp16 *dst, uint64_t count, uint32_t seed) {
    uint32_t s = seed;
    for (uint64_t i = 0; i < count; ++i) {
        float value = ((float) (next_rand_u32(&s) % 2001) / 2000.0f) - 0.5f;
        dst[i] = (__fp16) value;
    }
}

static void fill_f32(float *dst, uint64_t count, uint32_t seed) {
    uint32_t s = seed;
    for (uint64_t i = 0; i < count; ++i) {
        dst[i] = ((float) (next_rand_u32(&s) % 2001) / 2000.0f) - 0.5f;
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
        return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
    }
#endif
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static double elapsed_ms(uint64_t start_ns) {
    uint64_t end_ns = now_ns();
    if (end_ns < start_ns) {
        return 0.0;
    }
    return (double) (end_ns - start_ns) / 1000000.0;
}

static double checksum_f32(const float *data, uint64_t count) {
    double sum = 0.0;
    for (uint64_t i = 0; i < count; ++i) {
        sum += (double) data[i];
    }
    return sum;
}

static int scaled_iters(int base_iters) {
    if (g_iter_scale < 1) {
        return base_iters;
    }
    if (g_iter_scale > 1000000 / base_iters) {
        return 1000000;
    }
    return base_iters * g_iter_scale;
}

static void print_line(const char *name, int ok, int iters, uint64_t elems, double ms, double checksum) {
    printf("%-36s %-4s %7d %8" PRIu64 " %11.3f % .6e\n",
           name, ok ? "OK" : "FAIL", iters, elems, ms, checksum);
}

static int should_run(const char *name) {
    return g_case_filter == NULL || strcmp(g_case_filter, name) == 0;
}

static int run_scalar_query(const char *name, uint64_t (*fn)(void), int iters) {
    uint64_t value = 0;
    uint64_t start_ns = now_ns();
    for (int i = 0; i < iters; ++i) {
        value = fn();
    }
    print_line(name, 1, iters, 1, elapsed_ms(start_ns), (double) value);
    return 0;
}

static uint64_t wrap_has_sme(void) {
    return (uint64_t) sme_ai_has_sme();
}

static uint64_t wrap_has_sme2(void) {
    return (uint64_t) sme_ai_has_sme2();
}

static int case_has_sme(void) {
    return run_scalar_query("sme_ai_has_sme", wrap_has_sme, scaled_iters(1024));
}

static int case_has_sme2(void) {
    return run_scalar_query("sme_ai_has_sme2", wrap_has_sme2, scaled_iters(1024));
}

static int case_cntb(void) {
    return run_scalar_query("sme_ai_cntb", sme_ai_cntb, scaled_iters(1024));
}

static int case_cntw(void) {
    return run_scalar_query("sme_ai_cntw", sme_ai_cntw, scaled_iters(1024));
}

static int case_workspace_init(void) {
    int iters = scaled_iters(8);
    sme_ai_workspace *items = calloc((size_t) iters, sizeof(*items));
    double checksum = 0.0;
    uint64_t start_ns;
    int ok = items != NULL;
    if (!ok) {
        print_line("sme_ai_workspace_init", 0, iters, 0, 0.0, 0.0);
        return 1;
    }
    start_ns = now_ns();
    for (int i = 0; i < iters; ++i) {
        if (sme_ai_workspace_init(&items[i], 4, 64, 64) != SME_AI_OK) {
            ok = 0;
            break;
        }
        checksum += (double) items[i].cntw;
    }
    print_line("sme_ai_workspace_init", ok, iters, 0, elapsed_ms(start_ns), checksum);
    for (int i = 0; i < iters; ++i) {
        sme_ai_workspace_destroy(&items[i]);
    }
    free(items);
    return ok ? 0 : 1;
}

static int case_workspace_destroy(void) {
    int iters = scaled_iters(8);
    sme_ai_workspace *items = calloc((size_t) iters, sizeof(*items));
    uint64_t start_ns;
    int ok = items != NULL;
    if (!ok) {
        print_line("sme_ai_workspace_destroy", 0, iters, 0, 0.0, 0.0);
        return 1;
    }
    for (int i = 0; i < iters; ++i) {
        if (sme_ai_workspace_init(&items[i], 4, 64, 64) != SME_AI_OK) {
            ok = 0;
            break;
        }
    }
    start_ns = now_ns();
    for (int i = 0; i < iters; ++i) {
        sme_ai_workspace_destroy(&items[i]);
    }
    print_line("sme_ai_workspace_destroy", ok, iters, 0, elapsed_ms(start_ns), (double) iters);
    free(items);
    return ok ? 0 : 1;
}

static int case_packed_weight_init(void) {
    const uint64_t in_features = 64;
    const uint64_t out_features = 64;
    int iters = scaled_iters(8);
    __fp16 *weight = aligned_calloc_bytes(64, in_features * out_features * sizeof(__fp16));
    sme_ai_packed_linear_weight *items = calloc((size_t) iters, sizeof(*items));
    double checksum = 0.0;
    uint64_t start_ns;
    int ok = weight != NULL && items != NULL;
    if (!ok) {
        print_line("sme_ai_packed_linear_weight_init", 0, iters, 0, 0.0, 0.0);
        free(weight);
        free(items);
        return 1;
    }
    fill_fp16(weight, in_features * out_features, 111);
    start_ns = now_ns();
    for (int i = 0; i < iters; ++i) {
        if (sme_ai_packed_linear_weight_init(&items[i], weight, in_features, out_features) != SME_AI_OK) {
            ok = 0;
            break;
        }
        checksum += items[i].packed_weight != NULL ? checksum_f32(items[i].packed_weight, items[i].k_pad * items[i].n_pad) : 0.0;
    }
    print_line("sme_ai_packed_linear_weight_init", ok, iters, in_features * out_features, elapsed_ms(start_ns), checksum);
    for (int i = 0; i < iters; ++i) {
        sme_ai_packed_linear_weight_destroy(&items[i]);
    }
    free(weight);
    free(items);
    return ok ? 0 : 1;
}

static int case_packed_weight_destroy(void) {
    const uint64_t in_features = 64;
    const uint64_t out_features = 64;
    int iters = scaled_iters(8);
    __fp16 *weight = aligned_calloc_bytes(64, in_features * out_features * sizeof(__fp16));
    sme_ai_packed_linear_weight *items = calloc((size_t) iters, sizeof(*items));
    uint64_t start_ns;
    int ok = weight != NULL && items != NULL;
    if (!ok) {
        print_line("sme_ai_packed_linear_weight_destroy", 0, iters, 0, 0.0, 0.0);
        free(weight);
        free(items);
        return 1;
    }
    fill_fp16(weight, in_features * out_features, 112);
    for (int i = 0; i < iters; ++i) {
        if (sme_ai_packed_linear_weight_init(&items[i], weight, in_features, out_features) != SME_AI_OK) {
            ok = 0;
            break;
        }
    }
    start_ns = now_ns();
    for (int i = 0; i < iters; ++i) {
        sme_ai_packed_linear_weight_destroy(&items[i]);
    }
    print_line("sme_ai_packed_linear_weight_destroy", ok, iters, 0, elapsed_ms(start_ns), (double) iters);
    free(weight);
    free(items);
    return ok ? 0 : 1;
}

typedef enum {
    TENSOR_ADD = 0,
    TENSOR_MUL,
    TENSOR_RESIDUAL,
    TENSOR_ADD_BIAS,
    TENSOR_RELU,
    TENSOR_SIGMOID,
    TENSOR_TANH,
    TENSOR_GELU,
    TENSOR_SILU,
    TENSOR_SOFTMAX,
    TENSOR_SOFTMAX_MASKED,
    TENSOR_SOFTMAX_CAUSAL,
    TENSOR_LAYERNORM,
    TENSOR_RMSNORM,
} tensor_kind;

static int run_tensor_case(const char *name, tensor_kind kind) {
    enum { ROWS = 32, COLS = 32, COUNT = ROWS * COLS };
    int iters = scaled_iters(64);
    float *a = aligned_calloc_bytes(64, COUNT * sizeof(float));
    float *b = aligned_calloc_bytes(64, COUNT * sizeof(float));
    float *out = aligned_calloc_bytes(64, COUNT * sizeof(float));
    float *bias = aligned_calloc_bytes(64, COLS * sizeof(float));
    float *mask = aligned_calloc_bytes(64, COUNT * sizeof(float));
    uint64_t start_ns;
    int ok = a != NULL && b != NULL && out != NULL && bias != NULL && mask != NULL;

    if (!ok) {
        print_line(name, 0, iters, COUNT, 0.0, 0.0);
        goto cleanup;
    }
    fill_f32(a, COUNT, 201);
    fill_f32(b, COUNT, 202);
    fill_f32(bias, COLS, 203);
    for (uint64_t i = 0; i < COUNT; ++i) {
        mask[i] = (i % COLS == COLS - 1) ? -1000.0f : 0.0f;
    }

    start_ns = now_ns();
    for (int i = 0; i < iters; ++i) {
        switch (kind) {
            case TENSOR_ADD:
                sme_ai_add_f32(out, a, b, COUNT);
                break;
            case TENSOR_MUL:
                sme_ai_mul_f32(out, a, b, COUNT);
                break;
            case TENSOR_RESIDUAL:
                sme_ai_residual_add_f32(out, a, b, COUNT);
                break;
            case TENSOR_ADD_BIAS:
                sme_ai_add_bias_rowwise_f32(out, a, bias, ROWS, COLS);
                break;
            case TENSOR_RELU:
                memcpy(out, a, COUNT * sizeof(float));
                sme_ai_relu_inplace_f32(out, COUNT);
                break;
            case TENSOR_SIGMOID:
                memcpy(out, a, COUNT * sizeof(float));
                sme_ai_sigmoid_inplace_f32(out, COUNT);
                break;
            case TENSOR_TANH:
                memcpy(out, a, COUNT * sizeof(float));
                sme_ai_tanh_inplace_f32(out, COUNT);
                break;
            case TENSOR_GELU:
                memcpy(out, a, COUNT * sizeof(float));
                sme_ai_gelu_inplace_f32(out, COUNT);
                break;
            case TENSOR_SILU:
                memcpy(out, a, COUNT * sizeof(float));
                sme_ai_silu_inplace_f32(out, COUNT);
                break;
            case TENSOR_SOFTMAX:
                sme_ai_softmax_rowwise_f32(out, a, ROWS, COLS);
                break;
            case TENSOR_SOFTMAX_MASKED:
                sme_ai_softmax_masked_rowwise_f32(out, a, mask, ROWS, COLS);
                break;
            case TENSOR_SOFTMAX_CAUSAL:
                sme_ai_softmax_causal_rowwise_f32(out, a, ROWS, COLS);
                break;
            case TENSOR_LAYERNORM:
                sme_ai_layernorm_f32(out, a, bias, b, ROWS, COLS, 1e-5f);
                break;
            case TENSOR_RMSNORM:
                sme_ai_rmsnorm_f32(out, a, bias, ROWS, COLS, 1e-5f);
                break;
        }
        g_sink += out[(uint64_t) (i * 17) % COUNT];
    }
    print_line(name, ok, iters, COUNT, elapsed_ms(start_ns), checksum_f32(out, COUNT));

cleanup:
    free(a);
    free(b);
    free(out);
    free(bias);
    free(mask);
    return ok ? 0 : 1;
}

static int case_add_f32(void) { return run_tensor_case("sme_ai_add_f32", TENSOR_ADD); }
static int case_mul_f32(void) { return run_tensor_case("sme_ai_mul_f32", TENSOR_MUL); }
static int case_residual_add_f32(void) { return run_tensor_case("sme_ai_residual_add_f32", TENSOR_RESIDUAL); }
static int case_add_bias_rowwise_f32(void) { return run_tensor_case("sme_ai_add_bias_rowwise_f32", TENSOR_ADD_BIAS); }
static int case_relu_inplace_f32(void) { return run_tensor_case("sme_ai_relu_inplace_f32", TENSOR_RELU); }
static int case_sigmoid_inplace_f32(void) { return run_tensor_case("sme_ai_sigmoid_inplace_f32", TENSOR_SIGMOID); }
static int case_tanh_inplace_f32(void) { return run_tensor_case("sme_ai_tanh_inplace_f32", TENSOR_TANH); }
static int case_gelu_inplace_f32(void) { return run_tensor_case("sme_ai_gelu_inplace_f32", TENSOR_GELU); }
static int case_silu_inplace_f32(void) { return run_tensor_case("sme_ai_silu_inplace_f32", TENSOR_SILU); }
static int case_softmax_rowwise_f32(void) { return run_tensor_case("sme_ai_softmax_rowwise_f32", TENSOR_SOFTMAX); }
static int case_softmax_masked_f32(void) { return run_tensor_case("sme_ai_softmax_masked_rowwise_f32", TENSOR_SOFTMAX_MASKED); }
static int case_softmax_causal_f32(void) { return run_tensor_case("sme_ai_softmax_causal_rowwise_f32", TENSOR_SOFTMAX_CAUSAL); }
static int case_layernorm_f32(void) { return run_tensor_case("sme_ai_layernorm_f32", TENSOR_LAYERNORM); }
static int case_rmsnorm_f32(void) { return run_tensor_case("sme_ai_rmsnorm_f32", TENSOR_RMSNORM); }

static int case_embedding_lookup(void) {
    enum { VOCAB = 64, HIDDEN = 64, TOKENS = 16 };
    int iters = scaled_iters(64);
    __fp16 *table = aligned_calloc_bytes(64, VOCAB * HIDDEN * sizeof(__fp16));
    uint32_t token_ids[TOKENS];
    float *out = aligned_calloc_bytes(64, TOKENS * HIDDEN * sizeof(float));
    uint64_t start_ns;
    int ok = table != NULL && out != NULL;
    if (!ok) {
        print_line("sme_ai_embedding_lookup_f16f32", 0, iters, 0, 0.0, 0.0);
        free(table);
        free(out);
        return 1;
    }
    fill_fp16(table, VOCAB * HIDDEN, 301);
    for (uint32_t i = 0; i < TOKENS; ++i) {
        token_ids[i] = (i * 7u) % VOCAB;
    }
    start_ns = now_ns();
    for (int i = 0; i < iters; ++i) {
        if (sme_ai_embedding_lookup_f16f32(out, table, token_ids, TOKENS, VOCAB, HIDDEN) != SME_AI_OK) {
            ok = 0;
            break;
        }
        g_sink += out[(uint64_t) (i * 19) % (TOKENS * HIDDEN)];
    }
    print_line("sme_ai_embedding_lookup_f16f32", ok, iters, TOKENS * HIDDEN, elapsed_ms(start_ns), checksum_f32(out, TOKENS * HIDDEN));
    free(table);
    free(out);
    return ok ? 0 : 1;
}

typedef enum {
    POOL_MAX = 0,
    POOL_AVG,
} pool_kind;

static int run_pool_case(const char *name, pool_kind kind) {
    enum { BATCH = 1, CHANNELS = 4, IN_H = 16, IN_W = 16, OUT_H = 8, OUT_W = 8, COUNT = BATCH * CHANNELS * OUT_H * OUT_W };
    int iters = scaled_iters(64);
    float *input = aligned_calloc_bytes(64, BATCH * CHANNELS * IN_H * IN_W * sizeof(float));
    float *out = aligned_calloc_bytes(64, COUNT * sizeof(float));
    uint64_t start_ns;
    int ok = input != NULL && out != NULL;
    if (!ok) {
        print_line(name, 0, iters, 0, 0.0, 0.0);
        free(input);
        free(out);
        return 1;
    }
    fill_f32(input, BATCH * CHANNELS * IN_H * IN_W, 401);
    start_ns = now_ns();
    for (int i = 0; i < iters; ++i) {
        if (kind == POOL_MAX) {
            sme_ai_max_pool2d_nchw_f32(out, input, BATCH, CHANNELS, IN_H, IN_W, 2, 2, 2, 2);
        } else {
            sme_ai_avg_pool2d_nchw_f32(out, input, BATCH, CHANNELS, IN_H, IN_W, 2, 2, 2, 2);
        }
        g_sink += out[(uint64_t) (i * 23) % COUNT];
    }
    print_line(name, ok, iters, COUNT, elapsed_ms(start_ns), checksum_f32(out, COUNT));
    free(input);
    free(out);
    return 0;
}

static int case_max_pool2d(void) { return run_pool_case("sme_ai_max_pool2d_nchw_f32", POOL_MAX); }
static int case_avg_pool2d(void) { return run_pool_case("sme_ai_avg_pool2d_nchw_f32", POOL_AVG); }

static int run_linear_case(const char *name, linear_kind kind) {
    const uint64_t batch = 4;
    const uint64_t in_features = 64;
    const uint64_t out_features = 64;
    const uint64_t input_count = batch * in_features;
    const uint64_t weight_count = in_features * out_features;
    const uint64_t output_count = batch * out_features;
    int iters = scaled_iters(8);
    __fp16 *input = aligned_calloc_bytes(64, input_count * sizeof(__fp16));
    __fp16 *weight = aligned_calloc_bytes(64, weight_count * sizeof(__fp16));
    float *bias = aligned_calloc_bytes(64, out_features * sizeof(float));
    float *out = aligned_calloc_bytes(64, output_count * sizeof(float));
    sme_ai_workspace ws;
    sme_ai_packed_linear_weight packed;
    uint64_t start_ns;
    int ok = input != NULL && weight != NULL && bias != NULL && out != NULL;
    if (!ok) {
        print_line(name, 0, iters, 0, 0.0, 0.0);
        goto cleanup_no_ws;
    }
    memset(&ws, 0, sizeof(ws));
    memset(&packed, 0, sizeof(packed));
    fill_fp16(input, input_count, 501);
    fill_fp16(weight, weight_count, 502);
    fill_f32(bias, out_features, 503);
    if ((kind == LINEAR_SME_SIMPLE || kind == LINEAR_SME_PACKED || kind == LINEAR_RELU_SME || kind == LINEAR_GELU_SME || kind == LINEAR_SILU_SME) &&
        sme_ai_workspace_init(&ws, batch, in_features, out_features) != SME_AI_OK) {
        ok = 0;
        goto done;
    }
    if ((kind == LINEAR_SME_PACKED || kind == LINEAR_RELU_SME || kind == LINEAR_GELU_SME || kind == LINEAR_SILU_SME) &&
        sme_ai_packed_linear_weight_init(&packed, weight, in_features, out_features) != SME_AI_OK) {
        ok = 0;
        goto done;
    }
    start_ns = now_ns();
    for (int i = 0; i < iters && ok; ++i) {
        sme_ai_status st = SME_AI_OK;
        switch (kind) {
            case LINEAR_REF:
                st = sme_ai_linear_ref_f16f32(input, batch, in_features, weight, out_features, bias, out);
                break;
            case LINEAR_SME_SIMPLE:
                st = sme_ai_linear_sme_simple_f16f32(&ws, input, batch, in_features, weight, out_features, bias, out);
                break;
            case LINEAR_SME_PACKED:
                st = sme_ai_linear_sme_packed_f16f32(&ws, input, batch, &packed, bias, out);
                break;
            case LINEAR_RELU_REF:
                st = sme_ai_linear_relu_ref_f16f32(input, batch, in_features, weight, out_features, bias, out);
                break;
            case LINEAR_RELU_SME:
                st = sme_ai_linear_relu_sme_packed_f16f32(&ws, input, batch, &packed, bias, out);
                break;
            case LINEAR_GELU_REF:
                st = sme_ai_linear_gelu_ref_f16f32(input, batch, in_features, weight, out_features, bias, out);
                break;
            case LINEAR_GELU_SME:
                st = sme_ai_linear_gelu_sme_packed_f16f32(&ws, input, batch, &packed, bias, out);
                break;
            case LINEAR_SILU_REF:
                st = sme_ai_linear_silu_ref_f16f32(input, batch, in_features, weight, out_features, bias, out);
                break;
            case LINEAR_SILU_SME:
                st = sme_ai_linear_silu_sme_packed_f16f32(&ws, input, batch, &packed, bias, out);
                break;
        }
        ok = st == SME_AI_OK;
        g_sink += out[(uint64_t) (i * 29) % output_count];
    }
    print_line(name, ok, iters, output_count, elapsed_ms(start_ns), checksum_f32(out, output_count));
done:
    sme_ai_packed_linear_weight_destroy(&packed);
    sme_ai_workspace_destroy(&ws);
cleanup_no_ws:
    free(input);
    free(weight);
    free(bias);
    free(out);
    return ok ? 0 : 1;
}

static int case_linear_ref(void) { return run_linear_case("sme_ai_linear_ref_f16f32", LINEAR_REF); }
static int case_linear_sme_simple(void) { return run_linear_case("sme_ai_linear_sme_simple_f16f32", LINEAR_SME_SIMPLE); }
static int case_linear_sme_packed(void) { return run_linear_case("sme_ai_linear_sme_packed_f16f32", LINEAR_SME_PACKED); }
static int case_linear_relu_ref(void) { return run_linear_case("sme_ai_linear_relu_ref_f16f32", LINEAR_RELU_REF); }
static int case_linear_relu_sme(void) { return run_linear_case("sme_ai_linear_relu_sme_packed_f16f32", LINEAR_RELU_SME); }
static int case_linear_gelu_ref(void) { return run_linear_case("sme_ai_linear_gelu_ref_f16f32", LINEAR_GELU_REF); }
static int case_linear_gelu_sme(void) { return run_linear_case("sme_ai_linear_gelu_sme_packed_f16f32", LINEAR_GELU_SME); }
static int case_linear_silu_ref(void) { return run_linear_case("sme_ai_linear_silu_ref_f16f32", LINEAR_SILU_REF); }
static int case_linear_silu_sme(void) { return run_linear_case("sme_ai_linear_silu_sme_packed_f16f32", LINEAR_SILU_SME); }

static int run_mlp_case(const char *name, mlp_kind kind) {
    const uint64_t batch = 2;
    const uint64_t in_features = 64;
    const uint64_t hidden_features = 128;
    const uint64_t out_features = 64;
    const uint64_t input_count = batch * in_features;
    const uint64_t output_count = batch * out_features;
    int iters = scaled_iters(4);
    __fp16 *input = aligned_calloc_bytes(64, input_count * sizeof(__fp16));
    __fp16 *fc1 = aligned_calloc_bytes(64, in_features * hidden_features * sizeof(__fp16));
    __fp16 *fc2 = aligned_calloc_bytes(64, hidden_features * out_features * sizeof(__fp16));
    float *b1 = aligned_calloc_bytes(64, hidden_features * sizeof(float));
    float *b2 = aligned_calloc_bytes(64, out_features * sizeof(float));
    float *out = aligned_calloc_bytes(64, output_count * sizeof(float));
    sme_ai_workspace ws;
    sme_ai_packed_linear_weight p1;
    sme_ai_packed_linear_weight p2;
    uint64_t start_ns;
    int ok = input != NULL && fc1 != NULL && fc2 != NULL && b1 != NULL && b2 != NULL && out != NULL;
    if (!ok) {
        print_line(name, 0, iters, 0, 0.0, 0.0);
        goto cleanup_no_ws;
    }
    memset(&ws, 0, sizeof(ws));
    memset(&p1, 0, sizeof(p1));
    memset(&p2, 0, sizeof(p2));
    fill_fp16(input, input_count, 601);
    fill_fp16(fc1, in_features * hidden_features, 602);
    fill_fp16(fc2, hidden_features * out_features, 603);
    fill_f32(b1, hidden_features, 604);
    fill_f32(b2, out_features, 605);
    if (sme_ai_workspace_init(&ws, batch, hidden_features, hidden_features) != SME_AI_OK) {
        ok = 0;
        goto done;
    }
    if ((kind == MLP_RELU_SME || kind == MLP_GELU_SME) &&
        (sme_ai_packed_linear_weight_init(&p1, fc1, in_features, hidden_features) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&p2, fc2, hidden_features, out_features) != SME_AI_OK)) {
        ok = 0;
        goto done;
    }
    start_ns = now_ns();
    for (int i = 0; i < iters && ok; ++i) {
        sme_ai_status st = SME_AI_OK;
        if (kind == MLP_RELU_REF) {
            st = sme_ai_mlp_relu_ref_f16f32(&ws, input, batch, in_features, fc1, hidden_features, b1, fc2, out_features, b2, out);
        } else if (kind == MLP_RELU_SME) {
            st = sme_ai_mlp_relu_sme_packed_f16f32(&ws, input, batch, &p1, b1, &p2, b2, out);
        } else if (kind == MLP_GELU_REF) {
            st = sme_ai_mlp_gelu_ref_f16f32(&ws, input, batch, in_features, fc1, hidden_features, b1, fc2, out_features, b2, out);
        } else {
            st = sme_ai_mlp_gelu_sme_packed_f16f32(&ws, input, batch, &p1, b1, &p2, b2, out);
        }
        ok = st == SME_AI_OK;
        g_sink += out[(uint64_t) (i * 31) % output_count];
    }
    print_line(name, ok, iters, output_count, elapsed_ms(start_ns), checksum_f32(out, output_count));
done:
    sme_ai_packed_linear_weight_destroy(&p1);
    sme_ai_packed_linear_weight_destroy(&p2);
    sme_ai_workspace_destroy(&ws);
cleanup_no_ws:
    free(input);
    free(fc1);
    free(fc2);
    free(b1);
    free(b2);
    free(out);
    return ok ? 0 : 1;
}

static int case_mlp_relu_ref(void) { return run_mlp_case("sme_ai_mlp_relu_ref_f16f32", MLP_RELU_REF); }
static int case_mlp_relu_sme(void) { return run_mlp_case("sme_ai_mlp_relu_sme_packed_f16f32", MLP_RELU_SME); }
static int case_mlp_gelu_ref(void) { return run_mlp_case("sme_ai_mlp_gelu_ref_f16f32", MLP_GELU_REF); }
static int case_mlp_gelu_sme(void) { return run_mlp_case("sme_ai_mlp_gelu_sme_packed_f16f32", MLP_GELU_SME); }

static int run_swiglu_case(const char *name, swiglu_kind kind) {
    const uint64_t batch = 2;
    const uint64_t in_features = 64;
    const uint64_t hidden_features = 128;
    const uint64_t out_features = 64;
    const uint64_t input_count = batch * in_features;
    const uint64_t output_count = batch * out_features;
    int iters = scaled_iters(4);
    __fp16 *input = aligned_calloc_bytes(64, input_count * sizeof(__fp16));
    __fp16 *gate = aligned_calloc_bytes(64, in_features * hidden_features * sizeof(__fp16));
    __fp16 *up = aligned_calloc_bytes(64, in_features * hidden_features * sizeof(__fp16));
    __fp16 *down = aligned_calloc_bytes(64, hidden_features * out_features * sizeof(__fp16));
    float *bg = aligned_calloc_bytes(64, hidden_features * sizeof(float));
    float *bu = aligned_calloc_bytes(64, hidden_features * sizeof(float));
    float *bd = aligned_calloc_bytes(64, out_features * sizeof(float));
    float *out = aligned_calloc_bytes(64, output_count * sizeof(float));
    sme_ai_workspace ws;
    sme_ai_packed_linear_weight pg;
    sme_ai_packed_linear_weight pu;
    sme_ai_packed_linear_weight pd;
    uint64_t start_ns;
    int ok = input != NULL && gate != NULL && up != NULL && down != NULL && bg != NULL && bu != NULL && bd != NULL && out != NULL;
    if (!ok) {
        print_line(name, 0, iters, 0, 0.0, 0.0);
        goto cleanup_no_ws;
    }
    memset(&ws, 0, sizeof(ws));
    memset(&pg, 0, sizeof(pg));
    memset(&pu, 0, sizeof(pu));
    memset(&pd, 0, sizeof(pd));
    fill_fp16(input, input_count, 701);
    fill_fp16(gate, in_features * hidden_features, 702);
    fill_fp16(up, in_features * hidden_features, 703);
    fill_fp16(down, hidden_features * out_features, 704);
    fill_f32(bg, hidden_features, 705);
    fill_f32(bu, hidden_features, 706);
    fill_f32(bd, out_features, 707);
    if (sme_ai_workspace_init(&ws, batch, hidden_features, hidden_features) != SME_AI_OK) {
        ok = 0;
        goto done;
    }
    if (kind == SWIGLU_SME &&
        (sme_ai_packed_linear_weight_init(&pg, gate, in_features, hidden_features) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&pu, up, in_features, hidden_features) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&pd, down, hidden_features, out_features) != SME_AI_OK)) {
        ok = 0;
        goto done;
    }
    start_ns = now_ns();
    for (int i = 0; i < iters && ok; ++i) {
        sme_ai_status st;
        if (kind == SWIGLU_REF) {
            st = sme_ai_swiglu_ref_f16f32(&ws, input, batch, in_features, gate, hidden_features, bg, up, bu, down, out_features, bd, out);
        } else {
            st = sme_ai_swiglu_sme_packed_f16f32(&ws, input, batch, &pg, bg, &pu, bu, &pd, bd, out);
        }
        ok = st == SME_AI_OK;
        g_sink += out[(uint64_t) (i * 37) % output_count];
    }
    print_line(name, ok, iters, output_count, elapsed_ms(start_ns), checksum_f32(out, output_count));
done:
    sme_ai_packed_linear_weight_destroy(&pg);
    sme_ai_packed_linear_weight_destroy(&pu);
    sme_ai_packed_linear_weight_destroy(&pd);
    sme_ai_workspace_destroy(&ws);
cleanup_no_ws:
    free(input);
    free(gate);
    free(up);
    free(down);
    free(bg);
    free(bu);
    free(bd);
    free(out);
    return ok ? 0 : 1;
}

static int case_swiglu_ref(void) { return run_swiglu_case("sme_ai_swiglu_ref_f16f32", SWIGLU_REF); }
static int case_swiglu_sme(void) { return run_swiglu_case("sme_ai_swiglu_sme_packed_f16f32", SWIGLU_SME); }

static int run_attention_case(const char *name, attention_kind kind) {
    const uint64_t seq_len = 8;
    const uint64_t model_dim = 64;
    const uint64_t num_heads = 4;
    const uint64_t token_count = seq_len * model_dim;
    int iters = scaled_iters(2);
    __fp16 *input = aligned_calloc_bytes(64, token_count * sizeof(__fp16));
    __fp16 *q = aligned_calloc_bytes(64, model_dim * model_dim * sizeof(__fp16));
    __fp16 *k = aligned_calloc_bytes(64, model_dim * model_dim * sizeof(__fp16));
    __fp16 *v = aligned_calloc_bytes(64, model_dim * model_dim * sizeof(__fp16));
    __fp16 *o = aligned_calloc_bytes(64, model_dim * model_dim * sizeof(__fp16));
    float *bq = aligned_calloc_bytes(64, model_dim * sizeof(float));
    float *bk = aligned_calloc_bytes(64, model_dim * sizeof(float));
    float *bv = aligned_calloc_bytes(64, model_dim * sizeof(float));
    float *bo = aligned_calloc_bytes(64, model_dim * sizeof(float));
    float *out = aligned_calloc_bytes(64, token_count * sizeof(float));
    sme_ai_workspace ws;
    sme_ai_packed_linear_weight pq;
    sme_ai_packed_linear_weight pk;
    sme_ai_packed_linear_weight pv;
    sme_ai_packed_linear_weight po;
    uint64_t start_ns;
    int ok = input != NULL && q != NULL && k != NULL && v != NULL && o != NULL && bq != NULL && bk != NULL && bv != NULL && bo != NULL && out != NULL;
    if (!ok) {
        print_line(name, 0, iters, 0, 0.0, 0.0);
        goto cleanup_no_ws;
    }
    memset(&ws, 0, sizeof(ws));
    memset(&pq, 0, sizeof(pq));
    memset(&pk, 0, sizeof(pk));
    memset(&pv, 0, sizeof(pv));
    memset(&po, 0, sizeof(po));
    fill_fp16(input, token_count, 801);
    fill_fp16(q, model_dim * model_dim, 802);
    fill_fp16(k, model_dim * model_dim, 803);
    fill_fp16(v, model_dim * model_dim, 804);
    fill_fp16(o, model_dim * model_dim, 805);
    fill_f32(bq, model_dim, 806);
    fill_f32(bk, model_dim, 807);
    fill_f32(bv, model_dim, 808);
    fill_f32(bo, model_dim, 809);
    if (sme_ai_workspace_init(&ws, seq_len, model_dim, model_dim) != SME_AI_OK) {
        ok = 0;
        goto done;
    }
    if (kind == ATTN_SME &&
        (sme_ai_packed_linear_weight_init(&pq, q, model_dim, model_dim) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&pk, k, model_dim, model_dim) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&pv, v, model_dim, model_dim) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&po, o, model_dim, model_dim) != SME_AI_OK)) {
        ok = 0;
        goto done;
    }
    start_ns = now_ns();
    for (int i = 0; i < iters && ok; ++i) {
        sme_ai_status st;
        if (kind == ATTN_REF) {
            st = sme_ai_self_attention_ref_f16f32(&ws, input, seq_len, model_dim, num_heads, q, bq, k, bk, v, bv, o, bo, 1, out);
        } else {
            st = sme_ai_self_attention_sme_packed_f16f32(&ws, input, seq_len, model_dim, num_heads, &pq, bq, &pk, bk, &pv, bv, &po, bo, 1, out);
        }
        ok = st == SME_AI_OK;
        g_sink += out[(uint64_t) (i * 41) % token_count];
    }
    print_line(name, ok, iters, token_count, elapsed_ms(start_ns), checksum_f32(out, token_count));
done:
    sme_ai_packed_linear_weight_destroy(&pq);
    sme_ai_packed_linear_weight_destroy(&pk);
    sme_ai_packed_linear_weight_destroy(&pv);
    sme_ai_packed_linear_weight_destroy(&po);
    sme_ai_workspace_destroy(&ws);
cleanup_no_ws:
    free(input);
    free(q);
    free(k);
    free(v);
    free(o);
    free(bq);
    free(bk);
    free(bv);
    free(bo);
    free(out);
    return ok ? 0 : 1;
}

static int case_attention_ref(void) { return run_attention_case("sme_ai_self_attention_ref_f16f32", ATTN_REF); }
static int case_attention_sme(void) { return run_attention_case("sme_ai_self_attention_sme_packed_f16f32", ATTN_SME); }

static int run_conv_case(const char *name, conv_kind kind) {
    const uint64_t batch = 1;
    const uint64_t in_channels = 4;
    const uint64_t in_h = 12;
    const uint64_t in_w = 12;
    const uint64_t out_channels = 8;
    const uint64_t kernel = 3;
    const uint64_t out_h = 12;
    const uint64_t out_w = 12;
    const uint64_t input_count = batch * in_channels * in_h * in_w;
    const uint64_t weight_count = out_channels * in_channels * kernel * kernel;
    const uint64_t output_count = batch * out_channels * out_h * out_w;
    int iters = scaled_iters(2);
    __fp16 *input = aligned_calloc_bytes(64, input_count * sizeof(__fp16));
    __fp16 *weight = aligned_calloc_bytes(64, weight_count * sizeof(__fp16));
    float *bias = aligned_calloc_bytes(64, out_channels * sizeof(float));
    float *out = aligned_calloc_bytes(64, output_count * sizeof(float));
    sme_ai_workspace ws;
    uint64_t start_ns;
    int ok = input != NULL && weight != NULL && bias != NULL && out != NULL;
    if (!ok) {
        print_line(name, 0, iters, 0, 0.0, 0.0);
        goto cleanup_no_ws;
    }
    memset(&ws, 0, sizeof(ws));
    fill_fp16(input, input_count, 901);
    fill_fp16(weight, weight_count, 902);
    fill_f32(bias, out_channels, 903);
    if (kind == CONV_SME &&
        sme_ai_workspace_init(&ws, batch * out_h * out_w, in_channels * kernel * kernel, out_channels) != SME_AI_OK) {
        ok = 0;
        goto done;
    }
    start_ns = now_ns();
    for (int i = 0; i < iters && ok; ++i) {
        sme_ai_status st;
        if (kind == CONV_REF) {
            st = sme_ai_conv2d_ref_nchw_f16f32(input, batch, in_channels, in_h, in_w, weight, out_channels, kernel, kernel, 1, 1, 1, 1, bias, out);
        } else {
            st = sme_ai_conv2d_sme_im2col_nchw_f16f32(&ws, input, batch, in_channels, in_h, in_w, weight, out_channels, kernel, kernel, 1, 1, 1, 1, bias, out);
        }
        ok = st == SME_AI_OK;
        g_sink += out[(uint64_t) (i * 43) % output_count];
    }
    print_line(name, ok, iters, output_count, elapsed_ms(start_ns), checksum_f32(out, output_count));
done:
    sme_ai_workspace_destroy(&ws);
cleanup_no_ws:
    free(input);
    free(weight);
    free(bias);
    free(out);
    return ok ? 0 : 1;
}

static int case_conv_ref(void) { return run_conv_case("sme_ai_conv2d_ref_nchw_f16f32", CONV_REF); }
static int case_conv_sme(void) { return run_conv_case("sme_ai_conv2d_sme_im2col_nchw_f16f32", CONV_SME); }

static const case_entry k_cases[] = {
    {"sme_ai_has_sme", case_has_sme},
    {"sme_ai_has_sme2", case_has_sme2},
    {"sme_ai_cntb", case_cntb},
    {"sme_ai_cntw", case_cntw},
    {"sme_ai_workspace_init", case_workspace_init},
    {"sme_ai_workspace_destroy", case_workspace_destroy},
    {"sme_ai_packed_linear_weight_init", case_packed_weight_init},
    {"sme_ai_packed_linear_weight_destroy", case_packed_weight_destroy},
    {"sme_ai_add_f32", case_add_f32},
    {"sme_ai_mul_f32", case_mul_f32},
    {"sme_ai_residual_add_f32", case_residual_add_f32},
    {"sme_ai_add_bias_rowwise_f32", case_add_bias_rowwise_f32},
    {"sme_ai_relu_inplace_f32", case_relu_inplace_f32},
    {"sme_ai_sigmoid_inplace_f32", case_sigmoid_inplace_f32},
    {"sme_ai_tanh_inplace_f32", case_tanh_inplace_f32},
    {"sme_ai_gelu_inplace_f32", case_gelu_inplace_f32},
    {"sme_ai_silu_inplace_f32", case_silu_inplace_f32},
    {"sme_ai_softmax_rowwise_f32", case_softmax_rowwise_f32},
    {"sme_ai_softmax_masked_rowwise_f32", case_softmax_masked_f32},
    {"sme_ai_softmax_causal_rowwise_f32", case_softmax_causal_f32},
    {"sme_ai_layernorm_f32", case_layernorm_f32},
    {"sme_ai_rmsnorm_f32", case_rmsnorm_f32},
    {"sme_ai_embedding_lookup_f16f32", case_embedding_lookup},
    {"sme_ai_max_pool2d_nchw_f32", case_max_pool2d},
    {"sme_ai_avg_pool2d_nchw_f32", case_avg_pool2d},
    {"sme_ai_linear_ref_f16f32", case_linear_ref},
    {"sme_ai_linear_sme_simple_f16f32", case_linear_sme_simple},
    {"sme_ai_linear_sme_packed_f16f32", case_linear_sme_packed},
    {"sme_ai_linear_relu_ref_f16f32", case_linear_relu_ref},
    {"sme_ai_linear_relu_sme_packed_f16f32", case_linear_relu_sme},
    {"sme_ai_linear_gelu_ref_f16f32", case_linear_gelu_ref},
    {"sme_ai_linear_gelu_sme_packed_f16f32", case_linear_gelu_sme},
    {"sme_ai_linear_silu_ref_f16f32", case_linear_silu_ref},
    {"sme_ai_linear_silu_sme_packed_f16f32", case_linear_silu_sme},
    {"sme_ai_mlp_relu_ref_f16f32", case_mlp_relu_ref},
    {"sme_ai_mlp_relu_sme_packed_f16f32", case_mlp_relu_sme},
    {"sme_ai_mlp_gelu_ref_f16f32", case_mlp_gelu_ref},
    {"sme_ai_mlp_gelu_sme_packed_f16f32", case_mlp_gelu_sme},
    {"sme_ai_swiglu_ref_f16f32", case_swiglu_ref},
    {"sme_ai_swiglu_sme_packed_f16f32", case_swiglu_sme},
    {"sme_ai_self_attention_ref_f16f32", case_attention_ref},
    {"sme_ai_self_attention_sme_packed_f16f32", case_attention_sme},
    {"sme_ai_conv2d_ref_nchw_f16f32", case_conv_ref},
    {"sme_ai_conv2d_sme_im2col_nchw_f16f32", case_conv_sme},
};

static void print_usage(const char *argv0) {
    fprintf(stderr, "usage: %s [--case NAME] [--iter-scale N] [--no-header]\n", argv0);
}

static int parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--case") == 0) {
            if (++i >= argc) {
                return 0;
            }
            g_case_filter = argv[i];
        } else if (strcmp(argv[i], "--iter-scale") == 0) {
            char *end = NULL;
            long value;
            if (++i >= argc) {
                return 0;
            }
            value = strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || value < 1 || value > 1000000) {
                return 0;
            }
            g_iter_scale = (int) value;
        } else if (strcmp(argv[i], "--no-header") == 0) {
            g_no_header = 1;
        } else {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    int rc = 0;
    int matched = 0;

    if (!parse_args(argc, argv)) {
        print_usage(argv[0]);
        return 1;
    }
    if (!sme_ai_has_sme() || sme_ai_cntw() != 16) {
        fprintf(stderr, "This verifier expects SME1 with cntw=16.\n");
        return 2;
    }
    if (!g_no_header) {
        printf("ENV sme=%d sme2=%d cntb=%" PRIu64 " cntw=%" PRIu64 " iter_scale=%d\n",
               sme_ai_has_sme(), sme_ai_has_sme2(), sme_ai_cntb(), sme_ai_cntw(), g_iter_scale);
        printf("VERIFY_TIME\n");
        printf("%-36s %-4s %7s %8s %11s %13s\n", "CASE", "OK", "ITERS", "ELEMS", "TIME_ms", "CHECKSUM");
    }

    for (size_t i = 0; i < sizeof(k_cases) / sizeof(k_cases[0]); ++i) {
        if (should_run(k_cases[i].name)) {
            matched = 1;
            rc |= k_cases[i].fn();
        }
    }

    if (!matched) {
        fprintf(stderr, "unknown case: %s\n", g_case_filter);
        return 1;
    }
    if (!g_no_header) {
        printf("SINK %.6f\n", g_sink);
    }
    return rc;
}
