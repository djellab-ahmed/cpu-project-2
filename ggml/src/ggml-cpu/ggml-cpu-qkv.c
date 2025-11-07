#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"
#include "ggml-cpu/vec.h"

#include <math.h>

static inline void ggml_qkv_rope_apply(float * data,
                                       int head_dim,
                                       int rotary_dim,
                                       int rope_type,
                                       int pos,
                                       float rope_freq_base,
                                       float rope_freq_scale) {
    const int n_dims = rotary_dim > 0 ? rotary_dim : head_dim;

    if (n_dims == 0) {
        return;
    }

    GGML_ASSERT(n_dims <= head_dim);
    GGML_ASSERT((n_dims & 1) == 0);

    GGML_ASSERT((rope_type & GGML_ROPE_TYPE_MROPE) == 0);
    GGML_ASSERT((rope_type & GGML_ROPE_TYPE_VISION) == 0);

    const bool is_neox = (rope_type & GGML_ROPE_TYPE_NEOX) != 0;

    for (int i = 0; i < n_dims; i += 2) {
        const int pair_index = i / 2;
        const float exponent = -2.0f * pair_index / (float) n_dims;
        const float inv_freq = powf(rope_freq_base, exponent);
        const float theta = (float) pos * rope_freq_scale * inv_freq;
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);

        if (is_neox) {
            const int half = n_dims / 2;
            const int idx0 = pair_index;
            const int idx1 = pair_index + half;

            const float x0 = data[idx0];
            const float x1 = data[idx1];

            data[idx0] = x0 * cos_theta - x1 * sin_theta;
            data[idx1] = x0 * sin_theta + x1 * cos_theta;
        } else {
            const int idx0 = i;
            const int idx1 = i + 1;

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

    const int ith = params->ith;
    const int nth = params->nth;

    const int head_start = (n_head * ith) / nth;
    const int head_end = (n_head * (ith + 1)) / nth;

    if (head_start >= head_end) {
        return;
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

            vec_dot((int) d_model, &acc_q, 0, wq, 0, x_data, 0, 1);
            vec_dot((int) d_model, &acc_k, 0, wk, 0, x_data, 0, 1);
            vec_dot((int) d_model, &acc_v, 0, wv, 0, x_data, 0, 1);

            q_out[idx] = acc_q;
            k_out[idx] = acc_k;
            v_out[idx] = acc_v;
        }

        ggml_qkv_rope_apply(q_out + base, head_dim, rotary_dim, p->rope_type,
                            pos, rope_freq_base, rope_freq_scale);
        ggml_qkv_rope_apply(k_out + base, head_dim, rotary_dim, p->rope_type,
                            pos, rope_freq_base, rope_freq_scale);
    }
}
