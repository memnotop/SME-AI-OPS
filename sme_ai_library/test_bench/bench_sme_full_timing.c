#define _GNU_SOURCE

#include "sme_ai_ops.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
    MODE_BOTH = 0,
    MODE_REF,
    MODE_SME,
} run_mode;

static volatile float g_sink = 0.0f;
static run_mode g_mode = MODE_BOTH;
static const char *g_case_filter = NULL;
static int g_iter_scale = 1;

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

static uint64_t elapsed_ns(uint64_t start_ns) {
    uint64_t end_ns = now_ns();
    if (end_ns < start_ns) {
        return 0;
    }
    return end_ns - start_ns;
}

static double ns_to_ms(uint64_t elapsed_ns_value) {
    return (double) elapsed_ns_value / 1000000.0;
}

static double checksum_f32(const float *data, uint64_t count) {
    double sum = 0.0;
    for (uint64_t i = 0; i < count; ++i) {
        sum += (double) data[i];
    }
    return sum;
}

static int compare_arrays(const float *expected, const float *actual, uint64_t count, float abs_tol, float rel_tol, float *max_abs, float *max_rel) {
    *max_abs = 0.0f;
    *max_rel = 0.0f;
    for (uint64_t i = 0; i < count; ++i) {
        if (!isfinite(expected[i]) || !isfinite(actual[i])) {
            return 0;
        }
        float abs_err = fabsf(expected[i] - actual[i]);
        float denom = fabsf(expected[i]) > 1e-6f ? fabsf(expected[i]) : 1.0f;
        float rel_err = abs_err / denom;
        if (abs_err > *max_abs) {
            *max_abs = abs_err;
        }
        if (rel_err > *max_rel) {
            *max_rel = rel_err;
        }
        if (abs_err > abs_tol && rel_err > rel_tol) {
            return 0;
        }
    }
    return 1;
}

static int run_ref_mode(void) {
    return g_mode == MODE_BOTH || g_mode == MODE_REF;
}

static int run_sme_mode(void) {
    return g_mode == MODE_BOTH || g_mode == MODE_SME;
}

static const char *mode_name(void) {
    if (g_mode == MODE_REF) {
        return "ref";
    }
    if (g_mode == MODE_SME) {
        return "sme";
    }
    return "both";
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

static int should_run(const char *name) {
    return g_case_filter == NULL || strcmp(g_case_filter, name) == 0;
}

static double safe_speedup(double ref_ms, double sme_ms) {
    if (ref_ms <= 0.0 || sme_ms <= 0.0) {
        return 0.0;
    }
    return ref_ms / sme_ms;
}

static void print_result(const char *name, int iters, uint64_t elems, uint64_t ref_ns, uint64_t sme_ns, int ok, float max_abs, float max_rel, double checksum) {
    double ref_ms = ns_to_ms(ref_ns);
    double sme_ms = ns_to_ms(sme_ns);

    if (g_mode == MODE_BOTH) {
        printf("%-24s %7d %8" PRIu64 " %11.3f %11.3f %8.2fx %5s %10.3e %10.3e % .6e\n",
               name, iters, elems, ref_ms, sme_ms, safe_speedup(ref_ms, sme_ms), ok ? "pass" : "fail",
               (double) max_abs, (double) max_rel, checksum);
    } else {
        double ms = g_mode == MODE_REF ? ref_ms : sme_ms;
        printf("%-24s %-4s %7d %8" PRIu64 " %11.3f %11.3f %5s % .6e\n",
               name, mode_name(), iters, elems, ms, ms / (double) iters, ok ? "pass" : "fail", checksum);
    }
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--case NAME] [--mode both|ref|sme] [--iter-scale N]\n"
            "cases: linear_simple linear_packed linear_relu linear_gelu linear_silu mlp_relu mlp_gelu swiglu attention conv2d\n",
            argv0);
}

static int parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--case") == 0) {
            if (++i >= argc) {
                return 0;
            }
            g_case_filter = argv[i];
        } else if (strcmp(argv[i], "--mode") == 0) {
            if (++i >= argc) {
                return 0;
            }
            if (strcmp(argv[i], "both") == 0) {
                g_mode = MODE_BOTH;
            } else if (strcmp(argv[i], "ref") == 0) {
                g_mode = MODE_REF;
            } else if (strcmp(argv[i], "sme") == 0) {
                g_mode = MODE_SME;
            } else {
                return 0;
            }
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
        } else {
            return 0;
        }
    }
    return 1;
}

typedef enum {
    LINEAR_SIMPLE = 0,
    LINEAR_PACKED,
    LINEAR_RELU,
    LINEAR_GELU,
    LINEAR_SILU,
} linear_kind;

static int run_linear_case(const char *name, linear_kind kind) {
    const uint64_t batch = 8;
    const uint64_t in_features = 128;
    const uint64_t out_features = 128;
    const uint64_t input_count = batch * in_features;
    const uint64_t weight_count = in_features * out_features;
    const uint64_t output_count = batch * out_features;
    const int iters = scaled_iters(8);
    __fp16 *input = aligned_calloc_bytes(64, input_count * sizeof(__fp16));
    __fp16 *weight = aligned_calloc_bytes(64, weight_count * sizeof(__fp16));
    float *bias = aligned_calloc_bytes(64, out_features * sizeof(float));
    float *ref = aligned_calloc_bytes(64, output_count * sizeof(float));
    float *sme = aligned_calloc_bytes(64, output_count * sizeof(float));
    sme_ai_workspace ws;
    sme_ai_packed_linear_weight packed;
    uint64_t ref_ns = 0;
    uint64_t sme_ns = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int ok = 1;
    int rc = 1;

    if (input == NULL || weight == NULL || bias == NULL || ref == NULL || sme == NULL) {
        fprintf(stderr, "%s alloc failed\n", name);
        goto cleanup_no_ws;
    }

    memset(&ws, 0, sizeof(ws));
    memset(&packed, 0, sizeof(packed));
    fill_fp16(input, input_count, 1001);
    fill_fp16(weight, weight_count, 1002);
    fill_f32(bias, out_features, 1003);

    if (run_sme_mode()) {
        if (sme_ai_workspace_init(&ws, batch, in_features, out_features) != SME_AI_OK) {
            fprintf(stderr, "%s workspace failed\n", name);
            goto cleanup;
        }
        if (kind != LINEAR_SIMPLE &&
            sme_ai_packed_linear_weight_init(&packed, weight, in_features, out_features) != SME_AI_OK) {
            fprintf(stderr, "%s pack failed\n", name);
            goto cleanup;
        }
    }

    if (run_ref_mode()) {
        uint64_t start_ns;
        if ((kind == LINEAR_RELU && sme_ai_linear_relu_ref_f16f32(input, batch, in_features, weight, out_features, bias, ref) != SME_AI_OK) ||
            (kind == LINEAR_GELU && sme_ai_linear_gelu_ref_f16f32(input, batch, in_features, weight, out_features, bias, ref) != SME_AI_OK) ||
            (kind == LINEAR_SILU && sme_ai_linear_silu_ref_f16f32(input, batch, in_features, weight, out_features, bias, ref) != SME_AI_OK) ||
            ((kind == LINEAR_SIMPLE || kind == LINEAR_PACKED) && sme_ai_linear_ref_f16f32(input, batch, in_features, weight, out_features, bias, ref) != SME_AI_OK)) {
            fprintf(stderr, "%s ref warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if ((kind == LINEAR_RELU && sme_ai_linear_relu_ref_f16f32(input, batch, in_features, weight, out_features, bias, ref) != SME_AI_OK) ||
                (kind == LINEAR_GELU && sme_ai_linear_gelu_ref_f16f32(input, batch, in_features, weight, out_features, bias, ref) != SME_AI_OK) ||
                (kind == LINEAR_SILU && sme_ai_linear_silu_ref_f16f32(input, batch, in_features, weight, out_features, bias, ref) != SME_AI_OK) ||
                ((kind == LINEAR_SIMPLE || kind == LINEAR_PACKED) && sme_ai_linear_ref_f16f32(input, batch, in_features, weight, out_features, bias, ref) != SME_AI_OK)) {
                fprintf(stderr, "%s ref loop failed\n", name);
                goto cleanup;
            }
            g_sink += ref[(uint64_t) (i * 13) % output_count];
        }
        ref_ns = elapsed_ns(start_ns);
    }

    if (run_sme_mode()) {
        uint64_t start_ns;
        if ((kind == LINEAR_SIMPLE && sme_ai_linear_sme_simple_f16f32(&ws, input, batch, in_features, weight, out_features, bias, sme) != SME_AI_OK) ||
            (kind == LINEAR_PACKED && sme_ai_linear_sme_packed_f16f32(&ws, input, batch, &packed, bias, sme) != SME_AI_OK) ||
            (kind == LINEAR_RELU && sme_ai_linear_relu_sme_packed_f16f32(&ws, input, batch, &packed, bias, sme) != SME_AI_OK) ||
            (kind == LINEAR_GELU && sme_ai_linear_gelu_sme_packed_f16f32(&ws, input, batch, &packed, bias, sme) != SME_AI_OK) ||
            (kind == LINEAR_SILU && sme_ai_linear_silu_sme_packed_f16f32(&ws, input, batch, &packed, bias, sme) != SME_AI_OK)) {
            fprintf(stderr, "%s sme warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if ((kind == LINEAR_SIMPLE && sme_ai_linear_sme_simple_f16f32(&ws, input, batch, in_features, weight, out_features, bias, sme) != SME_AI_OK) ||
                (kind == LINEAR_PACKED && sme_ai_linear_sme_packed_f16f32(&ws, input, batch, &packed, bias, sme) != SME_AI_OK) ||
                (kind == LINEAR_RELU && sme_ai_linear_relu_sme_packed_f16f32(&ws, input, batch, &packed, bias, sme) != SME_AI_OK) ||
                (kind == LINEAR_GELU && sme_ai_linear_gelu_sme_packed_f16f32(&ws, input, batch, &packed, bias, sme) != SME_AI_OK) ||
                (kind == LINEAR_SILU && sme_ai_linear_silu_sme_packed_f16f32(&ws, input, batch, &packed, bias, sme) != SME_AI_OK)) {
                fprintf(stderr, "%s sme loop failed\n", name);
                goto cleanup;
            }
            g_sink += sme[(uint64_t) (i * 17) % output_count];
        }
        sme_ns = elapsed_ns(start_ns);
    }

    if (g_mode == MODE_BOTH) {
        ok = compare_arrays(ref, sme, output_count, 0.05f, 0.01f, &max_abs, &max_rel);
    }
    print_result(name, iters, output_count, ref_ns, sme_ns, ok, max_abs, max_rel, checksum_f32(run_sme_mode() ? sme : ref, output_count));
    rc = ok ? 0 : 1;

cleanup:
    sme_ai_packed_linear_weight_destroy(&packed);
    sme_ai_workspace_destroy(&ws);
cleanup_no_ws:
    free(input);
    free(weight);
    free(bias);
    free(ref);
    free(sme);
    return rc;
}

typedef enum {
    MLP_RELU = 0,
    MLP_GELU,
} mlp_kind;

static int run_mlp_case(const char *name, mlp_kind kind) {
    const uint64_t batch = 4;
    const uint64_t in_features = 128;
    const uint64_t hidden_features = 256;
    const uint64_t out_features = 128;
    const uint64_t input_count = batch * in_features;
    const uint64_t output_count = batch * out_features;
    const int iters = scaled_iters(4);
    __fp16 *input = aligned_calloc_bytes(64, input_count * sizeof(__fp16));
    __fp16 *fc1 = aligned_calloc_bytes(64, in_features * hidden_features * sizeof(__fp16));
    __fp16 *fc2 = aligned_calloc_bytes(64, hidden_features * out_features * sizeof(__fp16));
    float *b1 = aligned_calloc_bytes(64, hidden_features * sizeof(float));
    float *b2 = aligned_calloc_bytes(64, out_features * sizeof(float));
    float *ref = aligned_calloc_bytes(64, output_count * sizeof(float));
    float *sme = aligned_calloc_bytes(64, output_count * sizeof(float));
    sme_ai_workspace ws;
    sme_ai_packed_linear_weight p1;
    sme_ai_packed_linear_weight p2;
    uint64_t ref_ns = 0;
    uint64_t sme_ns = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int ok = 1;
    int rc = 1;

    if (input == NULL || fc1 == NULL || fc2 == NULL || b1 == NULL || b2 == NULL || ref == NULL || sme == NULL) {
        fprintf(stderr, "%s alloc failed\n", name);
        goto cleanup_no_ws;
    }

    memset(&ws, 0, sizeof(ws));
    memset(&p1, 0, sizeof(p1));
    memset(&p2, 0, sizeof(p2));
    fill_fp16(input, input_count, 2001);
    fill_fp16(fc1, in_features * hidden_features, 2002);
    fill_fp16(fc2, hidden_features * out_features, 2003);
    fill_f32(b1, hidden_features, 2004);
    fill_f32(b2, out_features, 2005);

    if (sme_ai_workspace_init(&ws, batch, hidden_features, hidden_features) != SME_AI_OK) {
        fprintf(stderr, "%s workspace failed\n", name);
        goto cleanup;
    }
    if (run_sme_mode() &&
        (sme_ai_packed_linear_weight_init(&p1, fc1, in_features, hidden_features) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&p2, fc2, hidden_features, out_features) != SME_AI_OK)) {
        fprintf(stderr, "%s pack failed\n", name);
        goto cleanup;
    }

    if (run_ref_mode()) {
        uint64_t start_ns;
        if ((kind == MLP_RELU && sme_ai_mlp_relu_ref_f16f32(&ws, input, batch, in_features, fc1, hidden_features, b1, fc2, out_features, b2, ref) != SME_AI_OK) ||
            (kind == MLP_GELU && sme_ai_mlp_gelu_ref_f16f32(&ws, input, batch, in_features, fc1, hidden_features, b1, fc2, out_features, b2, ref) != SME_AI_OK)) {
            fprintf(stderr, "%s ref warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if ((kind == MLP_RELU && sme_ai_mlp_relu_ref_f16f32(&ws, input, batch, in_features, fc1, hidden_features, b1, fc2, out_features, b2, ref) != SME_AI_OK) ||
                (kind == MLP_GELU && sme_ai_mlp_gelu_ref_f16f32(&ws, input, batch, in_features, fc1, hidden_features, b1, fc2, out_features, b2, ref) != SME_AI_OK)) {
                fprintf(stderr, "%s ref loop failed\n", name);
                goto cleanup;
            }
            g_sink += ref[(uint64_t) (i * 19) % output_count];
        }
        ref_ns = elapsed_ns(start_ns);
    }

    if (run_sme_mode()) {
        uint64_t start_ns;
        if ((kind == MLP_RELU && sme_ai_mlp_relu_sme_packed_f16f32(&ws, input, batch, &p1, b1, &p2, b2, sme) != SME_AI_OK) ||
            (kind == MLP_GELU && sme_ai_mlp_gelu_sme_packed_f16f32(&ws, input, batch, &p1, b1, &p2, b2, sme) != SME_AI_OK)) {
            fprintf(stderr, "%s sme warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if ((kind == MLP_RELU && sme_ai_mlp_relu_sme_packed_f16f32(&ws, input, batch, &p1, b1, &p2, b2, sme) != SME_AI_OK) ||
                (kind == MLP_GELU && sme_ai_mlp_gelu_sme_packed_f16f32(&ws, input, batch, &p1, b1, &p2, b2, sme) != SME_AI_OK)) {
                fprintf(stderr, "%s sme loop failed\n", name);
                goto cleanup;
            }
            g_sink += sme[(uint64_t) (i * 23) % output_count];
        }
        sme_ns = elapsed_ns(start_ns);
    }

    if (g_mode == MODE_BOTH) {
        ok = compare_arrays(ref, sme, output_count, 0.08f, 0.02f, &max_abs, &max_rel);
    }
    print_result(name, iters, output_count, ref_ns, sme_ns, ok, max_abs, max_rel, checksum_f32(run_sme_mode() ? sme : ref, output_count));
    rc = ok ? 0 : 1;

cleanup:
    sme_ai_packed_linear_weight_destroy(&p1);
    sme_ai_packed_linear_weight_destroy(&p2);
    sme_ai_workspace_destroy(&ws);
cleanup_no_ws:
    free(input);
    free(fc1);
    free(fc2);
    free(b1);
    free(b2);
    free(ref);
    free(sme);
    return rc;
}

static int run_swiglu_case(void) {
    const char *name = "swiglu";
    const uint64_t batch = 4;
    const uint64_t in_features = 128;
    const uint64_t hidden_features = 256;
    const uint64_t out_features = 128;
    const uint64_t input_count = batch * in_features;
    const uint64_t output_count = batch * out_features;
    const int iters = scaled_iters(4);
    __fp16 *input = aligned_calloc_bytes(64, input_count * sizeof(__fp16));
    __fp16 *gate = aligned_calloc_bytes(64, in_features * hidden_features * sizeof(__fp16));
    __fp16 *up = aligned_calloc_bytes(64, in_features * hidden_features * sizeof(__fp16));
    __fp16 *down = aligned_calloc_bytes(64, hidden_features * out_features * sizeof(__fp16));
    float *bg = aligned_calloc_bytes(64, hidden_features * sizeof(float));
    float *bu = aligned_calloc_bytes(64, hidden_features * sizeof(float));
    float *bd = aligned_calloc_bytes(64, out_features * sizeof(float));
    float *ref = aligned_calloc_bytes(64, output_count * sizeof(float));
    float *sme = aligned_calloc_bytes(64, output_count * sizeof(float));
    sme_ai_workspace ws;
    sme_ai_packed_linear_weight pg;
    sme_ai_packed_linear_weight pu;
    sme_ai_packed_linear_weight pd;
    uint64_t ref_ns = 0;
    uint64_t sme_ns = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int ok = 1;
    int rc = 1;

    if (input == NULL || gate == NULL || up == NULL || down == NULL || bg == NULL || bu == NULL || bd == NULL || ref == NULL || sme == NULL) {
        fprintf(stderr, "%s alloc failed\n", name);
        goto cleanup_no_ws;
    }

    memset(&ws, 0, sizeof(ws));
    memset(&pg, 0, sizeof(pg));
    memset(&pu, 0, sizeof(pu));
    memset(&pd, 0, sizeof(pd));
    fill_fp16(input, input_count, 3001);
    fill_fp16(gate, in_features * hidden_features, 3002);
    fill_fp16(up, in_features * hidden_features, 3003);
    fill_fp16(down, hidden_features * out_features, 3004);
    fill_f32(bg, hidden_features, 3005);
    fill_f32(bu, hidden_features, 3006);
    fill_f32(bd, out_features, 3007);

    if (sme_ai_workspace_init(&ws, batch, hidden_features, hidden_features) != SME_AI_OK) {
        fprintf(stderr, "%s workspace failed\n", name);
        goto cleanup;
    }
    if (run_sme_mode() &&
        (sme_ai_packed_linear_weight_init(&pg, gate, in_features, hidden_features) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&pu, up, in_features, hidden_features) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&pd, down, hidden_features, out_features) != SME_AI_OK)) {
        fprintf(stderr, "%s pack failed\n", name);
        goto cleanup;
    }

    if (run_ref_mode()) {
        uint64_t start_ns;
        if (sme_ai_swiglu_ref_f16f32(&ws, input, batch, in_features, gate, hidden_features, bg, up, bu, down, out_features, bd, ref) != SME_AI_OK) {
            fprintf(stderr, "%s ref warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if (sme_ai_swiglu_ref_f16f32(&ws, input, batch, in_features, gate, hidden_features, bg, up, bu, down, out_features, bd, ref) != SME_AI_OK) {
                fprintf(stderr, "%s ref loop failed\n", name);
                goto cleanup;
            }
            g_sink += ref[(uint64_t) (i * 29) % output_count];
        }
        ref_ns = elapsed_ns(start_ns);
    }

    if (run_sme_mode()) {
        uint64_t start_ns;
        if (sme_ai_swiglu_sme_packed_f16f32(&ws, input, batch, &pg, bg, &pu, bu, &pd, bd, sme) != SME_AI_OK) {
            fprintf(stderr, "%s sme warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if (sme_ai_swiglu_sme_packed_f16f32(&ws, input, batch, &pg, bg, &pu, bu, &pd, bd, sme) != SME_AI_OK) {
                fprintf(stderr, "%s sme loop failed\n", name);
                goto cleanup;
            }
            g_sink += sme[(uint64_t) (i * 31) % output_count];
        }
        sme_ns = elapsed_ns(start_ns);
    }

    if (g_mode == MODE_BOTH) {
        ok = compare_arrays(ref, sme, output_count, 0.08f, 0.02f, &max_abs, &max_rel);
    }
    print_result(name, iters, output_count, ref_ns, sme_ns, ok, max_abs, max_rel, checksum_f32(run_sme_mode() ? sme : ref, output_count));
    rc = ok ? 0 : 1;

cleanup:
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
    free(ref);
    free(sme);
    return rc;
}

static int run_attention_case(void) {
    const char *name = "attention";
    const uint64_t seq_len = 16;
    const uint64_t model_dim = 128;
    const uint64_t num_heads = 4;
    const uint64_t token_count = seq_len * model_dim;
    const int iters = scaled_iters(2);
    __fp16 *input = aligned_calloc_bytes(64, token_count * sizeof(__fp16));
    __fp16 *q = aligned_calloc_bytes(64, model_dim * model_dim * sizeof(__fp16));
    __fp16 *k = aligned_calloc_bytes(64, model_dim * model_dim * sizeof(__fp16));
    __fp16 *v = aligned_calloc_bytes(64, model_dim * model_dim * sizeof(__fp16));
    __fp16 *o = aligned_calloc_bytes(64, model_dim * model_dim * sizeof(__fp16));
    float *bq = aligned_calloc_bytes(64, model_dim * sizeof(float));
    float *bk = aligned_calloc_bytes(64, model_dim * sizeof(float));
    float *bv = aligned_calloc_bytes(64, model_dim * sizeof(float));
    float *bo = aligned_calloc_bytes(64, model_dim * sizeof(float));
    float *ref = aligned_calloc_bytes(64, token_count * sizeof(float));
    float *sme = aligned_calloc_bytes(64, token_count * sizeof(float));
    sme_ai_workspace ws;
    sme_ai_packed_linear_weight pq;
    sme_ai_packed_linear_weight pk;
    sme_ai_packed_linear_weight pv;
    sme_ai_packed_linear_weight po;
    uint64_t ref_ns = 0;
    uint64_t sme_ns = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int ok = 1;
    int rc = 1;

    if (input == NULL || q == NULL || k == NULL || v == NULL || o == NULL || bq == NULL || bk == NULL || bv == NULL || bo == NULL || ref == NULL || sme == NULL) {
        fprintf(stderr, "%s alloc failed\n", name);
        goto cleanup_no_ws;
    }

    memset(&ws, 0, sizeof(ws));
    memset(&pq, 0, sizeof(pq));
    memset(&pk, 0, sizeof(pk));
    memset(&pv, 0, sizeof(pv));
    memset(&po, 0, sizeof(po));
    fill_fp16(input, token_count, 4001);
    fill_fp16(q, model_dim * model_dim, 4002);
    fill_fp16(k, model_dim * model_dim, 4003);
    fill_fp16(v, model_dim * model_dim, 4004);
    fill_fp16(o, model_dim * model_dim, 4005);
    fill_f32(bq, model_dim, 4006);
    fill_f32(bk, model_dim, 4007);
    fill_f32(bv, model_dim, 4008);
    fill_f32(bo, model_dim, 4009);

    if (sme_ai_workspace_init(&ws, seq_len, model_dim, model_dim) != SME_AI_OK) {
        fprintf(stderr, "%s workspace failed\n", name);
        goto cleanup;
    }
    if (run_sme_mode() &&
        (sme_ai_packed_linear_weight_init(&pq, q, model_dim, model_dim) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&pk, k, model_dim, model_dim) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&pv, v, model_dim, model_dim) != SME_AI_OK ||
         sme_ai_packed_linear_weight_init(&po, o, model_dim, model_dim) != SME_AI_OK)) {
        fprintf(stderr, "%s pack failed\n", name);
        goto cleanup;
    }

    if (run_ref_mode()) {
        uint64_t start_ns;
        if (sme_ai_self_attention_ref_f16f32(&ws, input, seq_len, model_dim, num_heads, q, bq, k, bk, v, bv, o, bo, 1, ref) != SME_AI_OK) {
            fprintf(stderr, "%s ref warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if (sme_ai_self_attention_ref_f16f32(&ws, input, seq_len, model_dim, num_heads, q, bq, k, bk, v, bv, o, bo, 1, ref) != SME_AI_OK) {
                fprintf(stderr, "%s ref loop failed\n", name);
                goto cleanup;
            }
            g_sink += ref[(uint64_t) (i * 37) % token_count];
        }
        ref_ns = elapsed_ns(start_ns);
    }

    if (run_sme_mode()) {
        uint64_t start_ns;
        if (sme_ai_self_attention_sme_packed_f16f32(&ws, input, seq_len, model_dim, num_heads, &pq, bq, &pk, bk, &pv, bv, &po, bo, 1, sme) != SME_AI_OK) {
            fprintf(stderr, "%s sme warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if (sme_ai_self_attention_sme_packed_f16f32(&ws, input, seq_len, model_dim, num_heads, &pq, bq, &pk, bk, &pv, bv, &po, bo, 1, sme) != SME_AI_OK) {
                fprintf(stderr, "%s sme loop failed\n", name);
                goto cleanup;
            }
            g_sink += sme[(uint64_t) (i * 41) % token_count];
        }
        sme_ns = elapsed_ns(start_ns);
    }

    if (g_mode == MODE_BOTH) {
        ok = compare_arrays(ref, sme, token_count, 0.08f, 0.02f, &max_abs, &max_rel);
    }
    print_result(name, iters, token_count, ref_ns, sme_ns, ok, max_abs, max_rel, checksum_f32(run_sme_mode() ? sme : ref, token_count));
    rc = ok ? 0 : 1;

cleanup:
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
    free(ref);
    free(sme);
    return rc;
}

static int run_conv2d_case(void) {
    const char *name = "conv2d";
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
    const int iters = scaled_iters(4);
    __fp16 *input = aligned_calloc_bytes(64, input_count * sizeof(__fp16));
    __fp16 *weight = aligned_calloc_bytes(64, weight_count * sizeof(__fp16));
    float *bias = aligned_calloc_bytes(64, out_channels * sizeof(float));
    float *ref = aligned_calloc_bytes(64, output_count * sizeof(float));
    float *sme = aligned_calloc_bytes(64, output_count * sizeof(float));
    sme_ai_workspace ws;
    uint64_t ref_ns = 0;
    uint64_t sme_ns = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int ok = 1;
    int rc = 1;

    if (input == NULL || weight == NULL || bias == NULL || ref == NULL || sme == NULL) {
        fprintf(stderr, "%s alloc failed\n", name);
        goto cleanup_no_ws;
    }

    memset(&ws, 0, sizeof(ws));
    fill_fp16(input, input_count, 5001);
    fill_fp16(weight, weight_count, 5002);
    fill_f32(bias, out_channels, 5003);

    if (run_sme_mode() &&
        sme_ai_workspace_init(&ws, batch * out_h * out_w, in_channels * kernel * kernel, out_channels) != SME_AI_OK) {
        fprintf(stderr, "%s workspace failed\n", name);
        goto cleanup;
    }

    if (run_ref_mode()) {
        uint64_t start_ns;
        if (sme_ai_conv2d_ref_nchw_f16f32(input, batch, in_channels, in_h, in_w, weight, out_channels, kernel, kernel, 1, 1, 1, 1, bias, ref) != SME_AI_OK) {
            fprintf(stderr, "%s ref warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if (sme_ai_conv2d_ref_nchw_f16f32(input, batch, in_channels, in_h, in_w, weight, out_channels, kernel, kernel, 1, 1, 1, 1, bias, ref) != SME_AI_OK) {
                fprintf(stderr, "%s ref loop failed\n", name);
                goto cleanup;
            }
            g_sink += ref[(uint64_t) (i * 43) % output_count];
        }
        ref_ns = elapsed_ns(start_ns);
    }

    if (run_sme_mode()) {
        uint64_t start_ns;
        if (sme_ai_conv2d_sme_im2col_nchw_f16f32(&ws, input, batch, in_channels, in_h, in_w, weight, out_channels, kernel, kernel, 1, 1, 1, 1, bias, sme) != SME_AI_OK) {
            fprintf(stderr, "%s sme warmup failed\n", name);
            goto cleanup;
        }
        start_ns = now_ns();
        for (int i = 0; i < iters; ++i) {
            if (sme_ai_conv2d_sme_im2col_nchw_f16f32(&ws, input, batch, in_channels, in_h, in_w, weight, out_channels, kernel, kernel, 1, 1, 1, 1, bias, sme) != SME_AI_OK) {
                fprintf(stderr, "%s sme loop failed\n", name);
                goto cleanup;
            }
            g_sink += sme[(uint64_t) (i * 47) % output_count];
        }
        sme_ns = elapsed_ns(start_ns);
    }

    if (g_mode == MODE_BOTH) {
        ok = compare_arrays(ref, sme, output_count, 0.08f, 0.02f, &max_abs, &max_rel);
    }
    print_result(name, iters, output_count, ref_ns, sme_ns, ok, max_abs, max_rel, checksum_f32(run_sme_mode() ? sme : ref, output_count));
    rc = ok ? 0 : 1;

cleanup:
    sme_ai_workspace_destroy(&ws);
cleanup_no_ws:
    free(input);
    free(weight);
    free(bias);
    free(ref);
    free(sme);
    return rc;
}

int main(int argc, char **argv) {
    int rc = 0;

    if (!parse_args(argc, argv)) {
        print_usage(argv[0]);
        return 1;
    }
    if (!sme_ai_has_sme() || sme_ai_cntw() != 16) {
        fprintf(stderr, "This benchmark expects SME1 with cntw=16.\n");
        return 2;
    }

    printf("ENV sme=%d cntw=%" PRIu64 " timer=monotonic_raw mode=%s iter_scale=%d\n",
           sme_ai_has_sme(), sme_ai_cntw(), mode_name(), g_iter_scale);
    if (g_mode == MODE_BOTH) {
        printf("FULL_TIME_COMPARE\n");
        printf("%-24s %7s %8s %11s %11s %8s %5s %10s %10s %13s\n",
               "CASE", "ITERS", "ELEMS", "REF_ms", "SME_ms", "SPEEDUP", "CHECK", "MAX_ABS", "MAX_REL", "CHECKSUM");
    } else {
        printf("FULL_TIME_SINGLE\n");
        printf("%-24s %-4s %7s %8s %11s %11s %5s %13s\n",
               "CASE", "MODE", "ITERS", "ELEMS", "TOTAL_ms", "AVG_ms", "CHECK", "CHECKSUM");
    }

    if (should_run("linear_simple")) {
        rc |= run_linear_case("linear_simple", LINEAR_SIMPLE);
    }
    if (should_run("linear_packed")) {
        rc |= run_linear_case("linear_packed", LINEAR_PACKED);
    }
    if (should_run("linear_relu")) {
        rc |= run_linear_case("linear_relu", LINEAR_RELU);
    }
    if (should_run("linear_gelu")) {
        rc |= run_linear_case("linear_gelu", LINEAR_GELU);
    }
    if (should_run("linear_silu")) {
        rc |= run_linear_case("linear_silu", LINEAR_SILU);
    }
    if (should_run("mlp_relu")) {
        rc |= run_mlp_case("mlp_relu", MLP_RELU);
    }
    if (should_run("mlp_gelu")) {
        rc |= run_mlp_case("mlp_gelu", MLP_GELU);
    }
    if (should_run("swiglu")) {
        rc |= run_swiglu_case();
    }
    if (should_run("attention")) {
        rc |= run_attention_case();
    }
    if (should_run("conv2d")) {
        rc |= run_conv2d_case();
    }

    printf("SINK %.6f\n", g_sink);
    return rc;
}
