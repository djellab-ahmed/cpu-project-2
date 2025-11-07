#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"
#include "ggml-cpu/vec.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__linux__) || defined(__APPLE__)
#include <alloca.h>
#elif defined(_WIN32)
#include <malloc.h>
#define alloca _alloca
#endif

static inline int ggml_qkv_rotary_dim(int head_dim, int rotary_dim) {
    if (rotary_dim <= 0) {
        return head_dim;
    }
    return rotary_dim;
}

static inline void ggml_qkv_rope_make_twiddles(float * cos_tbl,
                                               float * sin_tbl,
                                               int rotary_dim,
                                               int pos,
                                               float rope_freq_base,
                                               float rope_freq_scale) {
    const int pair_count = rotary_dim / 2;
    for (int pair = 0; pair < pair_count; ++pair) {
        const float exponent = -2.0f * pair / (float) rotary_dim;
        const float inv_freq = powf(rope_freq_base, exponent);
        const float theta    = (float) pos * rope_freq_scale * inv_freq;
        cos_tbl[pair] = cosf(theta);
        sin_tbl[pair] = sinf(theta);
    }
}

static inline void ggml_qkv_rope_apply_precomputed(float * data,
                                                    int head_dim,
                                                    int rotary_dim,
                                                    int rope_type,
                                                    const float * cos_tbl,
                                                    const float * sin_tbl) {
    const int n_dims = ggml_qkv_rotary_dim(head_dim, rotary_dim);

    if (n_dims == 0) {
        return;
    }

    GGML_ASSERT(n_dims <= head_dim);
    GGML_ASSERT((n_dims & 1) == 0);

    GGML_ASSERT((rope_type & GGML_ROPE_TYPE_MROPE) == 0);
    GGML_ASSERT((rope_type & GGML_ROPE_TYPE_VISION) == 0);

    const bool is_neox = (rope_type & GGML_ROPE_TYPE_NEOX) != 0;
    const int half = n_dims / 2;

    for (int pair = 0; pair < half; ++pair) {
        const float cos_theta = cos_tbl[pair];
        const float sin_theta = sin_tbl[pair];

        if (is_neox) {
            const int idx0 = pair;
            const int idx1 = pair + half;

            const float x0 = data[idx0];
            const float x1 = data[idx1];

            data[idx0] = x0 * cos_theta - x1 * sin_theta;
            data[idx1] = x0 * sin_theta + x1 * cos_theta;
        } else {
            const int idx0 = pair * 2;
            const int idx1 = idx0 + 1;

            const float x0 = data[idx0];
            const float x1 = data[idx1];

            data[idx0] = x0 * cos_theta - x1 * sin_theta;
            data[idx1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

void ggml_compute_forward_qkv_mv_rope(const struct ggml_compute_params * params,
                                      struct ggml_tensor * dst) {
    const struct ggml_tensor * x = dst->src[0];
    const struct ggml_tensor * w = dst->src[1];

    GGML_ASSERT(x != NULL);
    GGML_ASSERT(w != NULL);

    const struct ggml_qkv_mv_rope_params * p =
        (const struct ggml_qkv_mv_rope_params *) dst->op_params;

    GGML_ASSERT(p != NULL);

    const int n_head = p->n_head;
    const int head_dim = p->head_dim;
    const int rotary_dim = p->rotary_dim;
    const int rope_type = p->rope_type;
    const int pos = p->pos;
    const float rope_freq_base = p->rope_freq_base;
    const float rope_freq_scale = p->rope_freq_scale;

    GGML_ASSERT(n_head > 0);
    GGML_ASSERT(head_dim > 0);

    const int64_t d_model = x->ne[0];
    const int64_t d_out = (int64_t) n_head * head_dim;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[1] == 1);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(w->ne[0] == d_model);
    GGML_ASSERT(w->ne[1] == 3 * d_out);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == 3 * d_out);
    GGML_ASSERT(dst->ne[1] == 1);

    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(w->type);
    GGML_ASSERT(traits != NULL);
    GGML_ASSERT(traits->vec_dot != NULL);

    ggml_vec_dot_t vec_dot = traits->vec_dot;

    const float * x_data = (const float *) x->data;
    const char  * w_data = (const char  *) w->data;
    float * out_data = (float *) dst->data;

    float * q_out = out_data;
    float * k_out = out_data + d_out;
    float * v_out = out_data + 2 * d_out;

    const size_t row_stride = w->nb[1];
    const size_t weight_stride = w->nb[0];
    const size_t x_stride = x->nb[0];

    const int ith = params->ith;
    const int nth = params->nth;

    const int head_start = (n_head * ith) / nth;
    const int head_end = (n_head * (ith + 1)) / nth;

    if (head_start >= head_end) {
        return;
    }

    const char * debug_env = getenv("GPTOSS_FUSE_QKV_ROPE_DEBUG");
    if (debug_env && atoi(debug_env) != 0 && params->ith == 0) {
        static atomic_bool printed = ATOMIC_VAR_INIT(false);
        bool expected = false;
        if (atomic_compare_exchange_strong(&printed, &expected, true)) {
            fprintf(stderr,
                    "[qkv+rope] fused decode fastpath enabled (n_head=%d, head_dim=%d, rotary_dim=%d, pos=%d)\n",
                    n_head, head_dim, ggml_qkv_rotary_dim(head_dim, rotary_dim), pos);
        }
    }

    const int rope_dims = ggml_qkv_rotary_dim(head_dim, rotary_dim);
    float * cos_tbl = NULL;
    float * sin_tbl = NULL;
    if (rope_dims > 0) {
        const int pair_count = rope_dims / 2;
        cos_tbl = (float *) alloca(sizeof(float) * (pair_count > 0 ? pair_count : 1));
        sin_tbl = (float *) alloca(sizeof(float) * (pair_count > 0 ? pair_count : 1));
        if (pair_count > 0) {
            ggml_qkv_rope_make_twiddles(cos_tbl, sin_tbl, rope_dims, pos, rope_freq_base, rope_freq_scale);
        }
    }

    for (int h = head_start; h < head_end; ++h) {
        const int64_t base = (int64_t) h * head_dim;

        for (int d = 0; d < head_dim; ++d) {
            const int64_t idx = base + d;

            float acc_q = 0.0f;
            float acc_k = 0.0f;
            float acc_v = 0.0f;

            const char * wq = w_data + (idx + 0 * d_out) * row_stride;
            const char * wk = w_data + (idx + 1 * d_out) * row_stride;
            const char * wv = w_data + (idx + 2 * d_out) * row_stride;

            vec_dot((int) d_model, &acc_q, 0, wq, weight_stride, x_data, x_stride, 1);
            vec_dot((int) d_model, &acc_k, 0, wk, weight_stride, x_data, x_stride, 1);
            vec_dot((int) d_model, &acc_v, 0, wv, weight_stride, x_data, x_stride, 1);

            q_out[idx] = acc_q;
            k_out[idx] = acc_k;
            v_out[idx] = acc_v;
        }

        if (rope_dims > 0 && cos_tbl != NULL && sin_tbl != NULL) {
            ggml_qkv_rope_apply_precomputed(q_out + base, head_dim, rotary_dim, rope_type, cos_tbl, sin_tbl);
            ggml_qkv_rope_apply_precomputed(k_out + base, head_dim, rotary_dim, rope_type, cos_tbl, sin_tbl);
        }
    }
}
