#include "sme_ai_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * 最小用户示例：
 * - 构造一个很小的 Linear 层；
 * - 先运行参考实现；
 * - 再运行 SME packed 实现；
 * - 比较两者输出误差。
 */

static void fill_input(__fp16 *input, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        const float v = (float)((int)(i % 7) - 3) * 0.125f;
        input[i] = (__fp16)v;
    }
}

static void fill_weight(__fp16 *weight, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        const float v = (float)((int)(i % 11) - 5) * 0.0625f;
        weight[i] = (__fp16)v;
    }
}

static void fill_bias(float *bias, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        bias[i] = (float)((int)(i % 5) - 2) * 0.03125f;
    }
}

static float max_abs_diff(const float *a, const float *b, uint64_t count) {
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < count; ++i) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
    }
    return max_abs;
}

int main(void) {
    enum {
        BATCH = 2,
        IN_FEATURES = 16,
        OUT_FEATURES = 16,
    };

    __fp16 input[BATCH * IN_FEATURES];
    __fp16 weight[IN_FEATURES * OUT_FEATURES];
    float bias[OUT_FEATURES];
    float ref[BATCH * OUT_FEATURES];
    float sme[BATCH * OUT_FEATURES];

    fill_input(input, BATCH * IN_FEATURES);
    fill_weight(weight, IN_FEATURES * OUT_FEATURES);
    fill_bias(bias, OUT_FEATURES);

    sme_ai_workspace ws;
    sme_ai_packed_linear_weight packed;

    sme_ai_status st = sme_ai_workspace_init(&ws, BATCH, IN_FEATURES, OUT_FEATURES);
    if (st != SME_AI_OK) {
        fprintf(stderr, "sme_ai_workspace_init failed: %d\n", (int)st);
        return 1;
    }

    st = sme_ai_packed_linear_weight_init(&packed, weight, IN_FEATURES, OUT_FEATURES);
    if (st != SME_AI_OK) {
        fprintf(stderr, "sme_ai_packed_linear_weight_init failed: %d\n", (int)st);
        sme_ai_workspace_destroy(&ws);
        return 1;
    }

    st = sme_ai_linear_ref_f16f32(input, BATCH, IN_FEATURES, weight, OUT_FEATURES, bias, ref);
    if (st != SME_AI_OK) {
        fprintf(stderr, "sme_ai_linear_ref_f16f32 failed: %d\n", (int)st);
        sme_ai_packed_linear_weight_destroy(&packed);
        sme_ai_workspace_destroy(&ws);
        return 1;
    }

    st = sme_ai_linear_sme_packed_f16f32(&ws, input, BATCH, &packed, bias, sme);
    if (st != SME_AI_OK) {
        fprintf(stderr, "sme_ai_linear_sme_packed_f16f32 failed: %d\n", (int)st);
        sme_ai_packed_linear_weight_destroy(&packed);
        sme_ai_workspace_destroy(&ws);
        return 1;
    }

    const float max_abs = max_abs_diff(ref, sme, BATCH * OUT_FEATURES);

    printf("DEMO_LINEAR max_abs=%0.8f\n", max_abs);
    printf("DEMO_LINEAR ref0=%0.8f sme0=%0.8f\n", ref[0], sme[0]);

    sme_ai_packed_linear_weight_destroy(&packed);
    sme_ai_workspace_destroy(&ws);

    if (max_abs > 0.001f) {
        fprintf(stderr, "DEMO_LINEAR FAIL: max_abs too large\n");
        return 1;
    }

    printf("DEMO_LINEAR PASS\n");
    return 0;
}
