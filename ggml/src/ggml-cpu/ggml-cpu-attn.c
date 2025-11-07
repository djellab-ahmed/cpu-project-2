#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ops.h"
#include "../../../src/gptoss-kv-view.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define GGML_TLS _Thread_local
#elif defined(_MSC_VER)
#define GGML_TLS __declspec(thread)
#else
#define GGML_TLS __thread
#endif

void flash_decode_q8_rowrow_avx2(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const struct gptoss_kv_view * kv_view);

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static bool gptoss_flash_decode_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        const char * env = getenv("GPTOSS_FLASH_DECODE");
        cached = (env == NULL) ? 1 : (atoi(env) != 0);
    }
    return cached != 0;
}

static GGML_TLS float * tls_q_buf = NULL;
static GGML_TLS size_t  tls_q_cap = 0;
static GGML_TLS float * tls_k_buf = NULL;
static GGML_TLS size_t  tls_k_cap = 0;
static GGML_TLS float * tls_v_buf = NULL;
static GGML_TLS size_t  tls_v_cap = 0;

static inline bool ggml_env_flag(const char * name) {
    const char * val = getenv(name);
    return val != NULL && val[0] != '\0' && val[0] != '0';
}

static inline int flash_dbg(void) {
    const char * env = getenv("GPTOSS_FLASH_DEBUG");
    return env && *env && *env != '0';
}

#ifndef GGML_ALIGNED_FREE_TAKES_SIZE
#define GGML_ALIGNED_FREE_TAKES_SIZE 1
#endif

#if GGML_ALIGNED_FREE_TAKES_SIZE
#define GGML_FREE_ALIGNED(p, sz) ggml_aligned_free((p), (sz))
#else
#define GGML_FREE_ALIGNED(p, sz) do { (void) (sz); ggml_aligned_free((p)); } while (0)
#endif

static void ensure_tls_buffer(float ** ptr, size_t * cap, size_t need) {
    if (*cap >= need) {
        return;
    }

    if (*ptr != NULL) {
        GGML_FREE_ALIGNED(*ptr, (*cap) * sizeof(float));
        *ptr = NULL;
    }

    if (need == 0) {
        *cap = 0;
        return;
    }

    *ptr = (float *) ggml_aligned_malloc(need * sizeof(float));
    GGML_ASSERT(*ptr != NULL);
    *cap = need;
}

static inline void load_row_to_f32(const void * src, enum ggml_type type, float * GGML_RESTRICT dst, int64_t n) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, (size_t) n * sizeof(float));
            break;
        case GGML_TYPE_F16:
            ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) src, dst, n);
            break;
        case GGML_TYPE_BF16:
            ggml_cpu_bf16_to_fp32((const ggml_bf16_t *) src, dst, n);
            break;
        default:
            GGML_ABORT("flash_attn_decode: unsupported dtype");
    }
}

static inline float dot_q_k_f32(const float * GGML_RESTRICT q, const float * GGML_RESTRICT k, int64_t hd) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= hd; i += 8) {
        __m256 vq = _mm256_loadu_ps(q + i);
        __m256 vk = _mm256_loadu_ps(k + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(vq, vk, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(vq, vk));
#endif
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < hd; ++i) {
        s += q[i] * k[i];
    }
    return s;
#else
    float s = 0.0f;
    for (int64_t i = 0; i < hd; ++i) {
        s += q[i] * k[i];
    }
    return s;
#endif
}

static inline size_t kv_token_head_offset(const struct gptoss_kv_view * view, int64_t stream, int64_t token, int64_t head) {
    size_t off = (size_t) token * view->stride_token_bytes + (size_t) head * view->stride_head_bytes;
    if (view->stride_stream_bytes != 0) {
        off += (size_t) stream * view->stride_stream_bytes;
    }
    return off;
}

static void flash_decode_q8_rowrow_scalar(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const struct gptoss_kv_view * kv_view) {
    struct ggml_tensor * q = dst->src[0];
    struct ggml_tensor * k = dst->src[1];

    float scale = 1.0f;
    memcpy(&scale, &dst->op_params[0], sizeof(float));

    const int64_t head_dim  = q->ne[0];
    const int64_t n_head    = q->ne[1];
    const int64_t n_batch   = q->ne[2];
    const int64_t n_stream  = q->ne[3] > 0 ? q->ne[3] : 1;
    const int64_t n_head_kv = k->ne[1];
    const int64_t n_kv      = k->ne[2];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const enum ggml_type q_type = q->type;

    GGML_ASSERT(q->ne[0] == kv_view->head_dim);
    GGML_ASSERT(kv_view->n_heads_kv == n_head_kv);
    GGML_ASSERT(n_batch == 1);
    GGML_ASSERT(n_stream == k->ne[3]);
    GGML_ASSERT(n_head % n_head_kv == 0);
    GGML_ASSERT(kv_view->interleaved);

    GGML_ASSERT(q->nb[0] == ggml_type_size(q_type));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const size_t q_nb1 = q->nb[1];
    const size_t q_nb2 = q->nb[2];
    const size_t q_nb3 = q->nb[3];

    const size_t y_nb1 = dst->nb[1];
    const size_t y_nb2 = dst->nb[2];
    const size_t y_nb3 = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t tasks = n_stream * n_batch * n_head;
    const int64_t task0 = (tasks * ith) / nth;
    const int64_t task1 = (tasks * (ith + 1)) / nth;
    if (task0 >= task1) {
        return;
    }

    const bool need_q_tmp = q_type != GGML_TYPE_F32;
    if (need_q_tmp) {
        ensure_tls_buffer(&tls_q_buf, &tls_q_cap, (size_t) head_dim);
    }

    int tile_tok = 0;
    memcpy(&tile_tok, &dst->op_params[4], sizeof(int));
    if (tile_tok < 32 || tile_tok > 2048) {
        tile_tok = 256;
    }
    const char * tile_env = getenv("GPTOSS_FLASH_TILE");
    if (tile_env) {
        int tmp = atoi(tile_env);
        if (tmp >= 32 && tmp <= 2048) {
            tile_tok = tmp;
        }
    }

    const int64_t head_ratio = n_head / n_head_kv;

    for (int64_t idx = task0; idx < task1; ++idx) {
        int64_t rem = idx;
        const int64_t stream = rem / (n_batch * n_head);
        rem -= stream * (n_batch * n_head);
        const int64_t batch  = rem / n_head;
        const int64_t head   = rem - batch * n_head;

        const int64_t kv_head = head / head_ratio;

        const char * q_ptr_raw = (const char *) q->data
                                 + head * q_nb1 + batch * q_nb2 + stream * q_nb3;
        const float * q_vec = NULL;
        if (q_type == GGML_TYPE_F32) {
            q_vec = (const float *) q_ptr_raw;
        } else {
            load_row_to_f32(q_ptr_raw, q_type, tls_q_buf, head_dim);
            q_vec = tls_q_buf;
        }

        float * y_ptr = (float *) ((char *) dst->data
                         + head * y_nb1 + batch * y_nb2 + stream * y_nb3);
        memset(y_ptr, 0, (size_t) head_dim * sizeof(float));

        float m = -INFINITY;
        float l = 0.0f;

        for (int64_t t0 = 0; t0 < n_kv; t0 += tile_tok) {
            const int64_t tile_n = (t0 + tile_tok <= n_kv) ? tile_tok : (n_kv - t0);

            for (int64_t ti = 0; ti < tile_n; ++ti) {
                const int64_t t = t0 + ti;

                size_t off = kv_token_head_offset(kv_view, stream, t, kv_head);
                const int8_t * Kq8 = (const int8_t *) (kv_view->base + off);
                const int8_t * Vq8 = Kq8 + head_dim;
                const ggml_fp16_t * sK_ptr = (const ggml_fp16_t *) (kv_view->base + off + 2 * (size_t) head_dim);
                const ggml_fp16_t * sV_ptr = sK_ptr + 1;

                const float sK = ggml_fp16_to_fp32(*sK_ptr);
                const float sV = ggml_fp16_to_fp32(*sV_ptr);

                float sum = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    sum += q_vec[d] * ((float) Kq8[d] * sK);
                }
                const float s = sum * scale;

                const float m_new = fmaxf(m, s);
                const float alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
                const float beta  = expf(s - m_new);

                for (int64_t d = 0; d < head_dim; ++d) {
                    const float v_val = (float) Vq8[d] * sV;
                    y_ptr[d] = alpha * y_ptr[d] + beta * v_val;
                }

                l = alpha * l + beta;
                m = m_new;
            }
        }

        if (l > 0.0f) {
            const float inv_l = 1.0f / l;
            for (int64_t i = 0; i < head_dim; ++i) {
                y_ptr[i] *= inv_l;
            }
        }
    }
}

static void ggml_compute_forward_flash_attn_decode_cpu_fp16(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    if (!gptoss_flash_decode_enabled()) {
        GGML_ABORT("flash_attn_decode disabled; dispatch should have avoided this path");
    }

    struct ggml_tensor * q = dst->src[0];
    struct ggml_tensor * k = dst->src[1];
    struct ggml_tensor * v = dst->src[2];

    float scale = 1.0f;
    memcpy(&scale, &dst->op_params[0], sizeof(float));

    // tensor layouts after graph permutes:
    // q: [D, n_head,    n_batch, n_stream]
    // k: [D, n_head_kv, n_kv,    n_stream]
    // v: [D, n_head_kv, n_kv,    n_stream]
    const int64_t head_dim  = q->ne[0];
    const int64_t n_head    = q->ne[1];
    const int64_t n_batch   = q->ne[2];
    const int64_t n_stream  = q->ne[3] > 0 ? q->ne[3] : 1;
    const int64_t n_head_kv = k->ne[1];
    const int64_t n_kv      = k->ne[2];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const enum ggml_type q_type = q->type;
    const enum ggml_type k_type = k->type;
    const enum ggml_type v_type = v->type;

    GGML_ASSERT(q->ne[0] == k->ne[0] && q->ne[0] == v->ne[0]);
    GGML_ASSERT(k->ne[1] == v->ne[1]);
    GGML_ASSERT(k->ne[2] == v->ne[2]);
    GGML_ASSERT(k->ne[3] == v->ne[3]);
    GGML_ASSERT(n_stream == k->ne[3]);
    GGML_ASSERT(n_head % n_head_kv == 0);
    GGML_ASSERT(n_batch == 1);

    GGML_ASSERT(q->nb[0] == ggml_type_size(q_type));
    GGML_ASSERT(k->nb[0] == ggml_type_size(k_type));
    GGML_ASSERT(v->nb[0] == ggml_type_size(v_type));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const size_t q_nb1 = q->nb[1];
    const size_t q_nb2 = q->nb[2];
    const size_t q_nb3 = q->nb[3];

    const size_t k_nb1 = k->nb[1];
    const size_t k_nb2 = k->nb[2];
    const size_t k_nb3 = k->nb[3];

    const size_t v_nb1 = v->nb[1];
    const size_t v_nb2 = v->nb[2];
    const size_t v_nb3 = v->nb[3];

    const size_t y_nb1 = dst->nb[1];
    const size_t y_nb2 = dst->nb[2];
    const size_t y_nb3 = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t tasks = n_stream * n_batch * n_head;
    const int64_t task0 = (tasks * ith) / nth;
    const int64_t task1 = (tasks * (ith + 1)) / nth;
    if (task0 >= task1) {
        return;
    }

    const bool need_q_tmp = q_type != GGML_TYPE_F32;
    const bool need_k_tmp = k_type != GGML_TYPE_F32;
    const bool need_v_tmp = v_type != GGML_TYPE_F32;

    if (need_q_tmp) {
        ensure_tls_buffer(&tls_q_buf, &tls_q_cap, (size_t) head_dim);
    }
    if (need_k_tmp) {
        ensure_tls_buffer(&tls_k_buf, &tls_k_cap, (size_t) head_dim);
    }
    if (need_v_tmp) {
        ensure_tls_buffer(&tls_v_buf, &tls_v_cap, (size_t) head_dim);
    }

    int tile_tok = 0;
    memcpy(&tile_tok, &dst->op_params[4], sizeof(int));
    if (tile_tok < 32 || tile_tok > 2048) {
        tile_tok = 256;
    }
    const char * tile_env = getenv("GPTOSS_FLASH_TILE");
    if (tile_env) {
        int tmp = atoi(tile_env);
        if (tmp >= 32 && tmp <= 2048) {
            tile_tok = tmp;
        }
    }

    if (params->ith == 0 && flash_dbg()) {
        static int once;
        if (!once++) {
            fprintf(stderr, "[flash-decode] online-softmax fastpath enabled (tile=%d)\n", tile_tok);
        }
    }

    const int64_t head_ratio = n_head / n_head_kv;

    for (int64_t idx = task0; idx < task1; ++idx) {
        int64_t rem = idx;
        const int64_t stream = rem / (n_batch * n_head);
        rem -= stream * (n_batch * n_head);
        const int64_t batch  = rem / n_head;
        const int64_t head   = rem - batch * n_head;

        const int64_t kv_head = head / head_ratio;

        const char * q_ptr_raw = (const char *) q->data
                                 + head * q_nb1 + batch * q_nb2 + stream * q_nb3;
        const float * q_vec = NULL;
        if (q_type == GGML_TYPE_F32) {
            q_vec = (const float *) q_ptr_raw;
        } else {
            load_row_to_f32(q_ptr_raw, q_type, tls_q_buf, head_dim);
            q_vec = tls_q_buf;
        }

        float * y_ptr = (float *) ((char *) dst->data
                         + head * y_nb1 + batch * y_nb2 + stream * y_nb3);
        memset(y_ptr, 0, (size_t) head_dim * sizeof(float));

        float m = -INFINITY;
        float l = 0.0f;

        for (int64_t t0 = 0; t0 < n_kv; t0 += tile_tok) {
            const int64_t tile_n = (t0 + tile_tok <= n_kv) ? tile_tok : (n_kv - t0);

#if defined(__AVX2__)
            if (t0 + tile_n < n_kv) {
                const char * k_next = (const char *) k->data
                                     + kv_head * k_nb1 + (t0 + tile_n) * k_nb2 + stream * k_nb3;
                const char * v_next = (const char *) v->data
                                     + kv_head * v_nb1 + (t0 + tile_n) * v_nb2 + stream * v_nb3;
                _mm_prefetch(k_next, _MM_HINT_T0);
                _mm_prefetch(v_next, _MM_HINT_T0);
            }
#endif

            for (int64_t ti = 0; ti < tile_n; ++ti) {
                const int64_t t = t0 + ti;

                const char * k_ptr_raw = (const char *) k->data
                                         + kv_head * k_nb1 + t * k_nb2 + stream * k_nb3;
                const float * k_vec = NULL;
                if (k_type == GGML_TYPE_F32) {
                    k_vec = (const float *) k_ptr_raw;
                } else {
                    load_row_to_f32(k_ptr_raw, k_type, tls_k_buf, head_dim);
                    k_vec = tls_k_buf;
                }

                float s = dot_q_k_f32(q_vec, k_vec, head_dim) * scale;

                const float m_new = fmaxf(m, s);
                const float alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
                const float beta  = expf(s - m_new);

                const char * v_ptr_raw = (const char *) v->data
                                         + kv_head * v_nb1 + t * v_nb2 + stream * v_nb3;
                const float * v_vec = NULL;
                if (v_type == GGML_TYPE_F32) {
                    v_vec = (const float *) v_ptr_raw;
                } else {
                    load_row_to_f32(v_ptr_raw, v_type, tls_v_buf, head_dim);
                    v_vec = tls_v_buf;
                }
                
#if defined(__AVX2__)
                {
                    __m256 va = _mm256_set1_ps(alpha);
                    __m256 vb = _mm256_set1_ps(beta);
                    int64_t i = 0;
                    for (; i + 8 <= head_dim; i += 8) {
                        __m256 u  = _mm256_loadu_ps(y_ptr + i);
                        __m256 vv = _mm256_loadu_ps(v_vec + i);
#if defined(__FMA__)
                        u = _mm256_fmadd_ps(vb, vv, _mm256_mul_ps(va, u));
#else
                        u = _mm256_add_ps(_mm256_mul_ps(va, u), _mm256_mul_ps(vb, vv));
#endif
                        _mm256_storeu_ps(y_ptr + i, u);
                    }
                    for (; i < head_dim; ++i) {
                        y_ptr[i] = alpha * y_ptr[i] + beta * v_vec[i];
                    }
                }
#else
                for (int64_t i = 0; i < head_dim; ++i) {
                    y_ptr[i] = alpha * y_ptr[i] + beta * v_vec[i];
                }
#endif

                l = alpha * l + beta;
                m = m_new;
            }
        }

        if (l > 0.0f) {
#if defined(__AVX2__)
            {
                const float inv_l = 1.0f / l;
                __m256 vinv = _mm256_set1_ps(inv_l);
                int64_t i = 0;
                for (; i + 8 <= head_dim; i += 8) {
                    __m256 u = _mm256_loadu_ps(y_ptr + i);
                    _mm256_storeu_ps(y_ptr + i, _mm256_mul_ps(u, vinv));
                }
                for (; i < head_dim; ++i) {
                    y_ptr[i] *= inv_l;
                }
            }
#else
            const float inv_l = 1.0f / l;
            for (int64_t i = 0; i < head_dim; ++i) {
                y_ptr[i] *= inv_l;
            }
#endif
        }
    }
}

void ggml_compute_forward_flash_attn_decode_cpu(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct gptoss_kv_view * kv_view = (const struct gptoss_kv_view *) dst->extra;
    const int32_t kv_dtype = ggml_get_op_params_i32(dst, 2);

    if (kv_view != NULL &&
        kv_dtype == GPTOSS_KV_Q8_ROWROW &&
        kv_view->dtype_k == GPTOSS_KV_Q8_ROWROW &&
        kv_view->dtype_v == GPTOSS_KV_Q8_ROWROW) {
        if (GGML_UNLIKELY(getenv("GPTOSS_KVQ8_DEBUG"))) {
            static _Atomic int once = 0;
            if (atomic_exchange(&once, 1) == 0) {
                int tile_tok = 0;
                memcpy(&tile_tok, &dst->op_params[4], sizeof(int));
                if (tile_tok < 32 || tile_tok > 2048) {
                    tile_tok = 256;
                }
                GGML_LOG_INFO("flash-decode: Q8 row,row path engaged (tile=%d)\n", tile_tok);
            }
        }
        if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) {
            flash_decode_q8_rowrow_avx2(params, dst, kv_view);
        } else {
            flash_decode_q8_rowrow_scalar(params, dst, kv_view);
        }
        return;
    }

    ggml_compute_forward_flash_attn_decode_cpu_fp16(params, dst);
}

