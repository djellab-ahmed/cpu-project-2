#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ops.h"
#include "../../../src/gptoss-kv-view.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__AVX2__) || !defined(__FMA__)
#error "ggml-cpu-attn-q8-avx2.c requires AVX2 and FMA"
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define GGML_TLS _Thread_local
#elif defined(_MSC_VER)
#define GGML_TLS __declspec(thread)
#else
#define GGML_TLS __thread
#endif

static GGML_TLS float * tls_q_buf = NULL;
static GGML_TLS size_t  tls_q_cap = 0;

#ifndef GGML_ALIGNED_FREE_TAKES_SIZE
#define GGML_ALIGNED_FREE_TAKES_SIZE 1
#endif

#if GGML_ALIGNED_FREE_TAKES_SIZE
#define GGML_FREE_ALIGNED(p, sz) ggml_aligned_free((p), (sz))
#else
#define GGML_FREE_ALIGNED(p, sz) do { (void) (sz); ggml_aligned_free((p)); } while (0)
#endif

static inline void ensure_tls_buffer(float ** ptr, size_t * cap, size_t need) {
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

static inline size_t kv_token_head_offset(const struct gptoss_kv_view * view, int64_t stream, int64_t token, int64_t head) {
    size_t off = (size_t) token * view->stride_token_bytes + (size_t) head * view->stride_head_bytes;
    if (view->stride_stream_bytes != 0) {
        off += (size_t) stream * view->stride_stream_bytes;
    }
    return off;
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}

static inline float dot_q8_rowrow_avx2(const float * GGML_RESTRICT q, const int8_t * GGML_RESTRICT k, float sK, int hd) {
    const __m256 scale = _mm256_set1_ps(sK);
    __m256 acc = _mm256_setzero_ps();
    int i = 0;

    for (; i + 16 <= hd; i += 16) {
        const __m128i bytes = _mm_loadu_si128((const __m128i *) (k + i));
        const __m256i i16 = _mm256_cvtepi8_epi16(bytes);
        const __m128i lo16 = _mm256_castsi256_si128(i16);
        const __m128i hi16 = _mm256_extracti128_si256(i16, 1);

        const __m256 fk0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo16)), scale);
        const __m256 fk1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi16)), scale);

        const __m256 fq0 = _mm256_loadu_ps(q + i);
        const __m256 fq1 = _mm256_loadu_ps(q + i + 8);

        acc = _mm256_fmadd_ps(fq0, fk0, acc);
        acc = _mm256_fmadd_ps(fq1, fk1, acc);
    }

    float sum = hsum256_ps(acc);
    for (; i < hd; ++i) {
        sum += q[i] * ((float) k[i] * sK);
    }

    return sum;
}

static inline void scale_row_f32_avx2(float * GGML_RESTRICT data, float alpha, int hd) {
    if (alpha == 1.0f) {
        return;
    }

    const __m256 a = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i + 16 <= hd; i += 16) {
        __m256 y0 = _mm256_loadu_ps(data + i);
        __m256 y1 = _mm256_loadu_ps(data + i + 8);
        _mm256_storeu_ps(data + i,     _mm256_mul_ps(y0, a));
        _mm256_storeu_ps(data + i + 8, _mm256_mul_ps(y1, a));
    }
    for (; i < hd; ++i) {
        data[i] *= alpha;
    }
}

static inline void axpy_q8_rowrow_avx2(float * GGML_RESTRICT o, float p, const int8_t * GGML_RESTRICT v, float sV, int hd) {
    if (p == 0.0f) {
        return;
    }

    const __m256 scale = _mm256_set1_ps(p * sV);
    int i = 0;
    for (; i + 16 <= hd; i += 16) {
        const __m128i bytes = _mm_loadu_si128((const __m128i *) (v + i));
        const __m256i i16 = _mm256_cvtepi8_epi16(bytes);
        const __m128i lo16 = _mm256_castsi256_si128(i16);
        const __m128i hi16 = _mm256_extracti128_si256(i16, 1);

        const __m256 fv0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo16)), scale);
        const __m256 fv1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi16)), scale);

        __m256 yo = _mm256_loadu_ps(o + i);
        __m256 yo1 = _mm256_loadu_ps(o + i + 8);
        yo  = _mm256_add_ps(yo,  fv0);
        yo1 = _mm256_add_ps(yo1, fv1);
        _mm256_storeu_ps(o + i,     yo);
        _mm256_storeu_ps(o + i + 8, yo1);
    }

    for (; i < hd; ++i) {
        o[i] += p * ((float) v[i] * sV);
    }
}

void flash_decode_q8_rowrow_avx2(
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

                const float dot = dot_q8_rowrow_avx2(q_vec, Kq8, sK, (int) head_dim);
                const float s = dot * scale;

                const float m_new = fmaxf(m, s);
                const float alpha = isfinite(m) ? expf(m - m_new) : 0.0f;
                const float beta  = expf(s - m_new);

                if (alpha != 1.0f) {
                    scale_row_f32_avx2(y_ptr, alpha, (int) head_dim);
                }
                axpy_q8_rowrow_avx2(y_ptr, beta, Vq8, sV, (int) head_dim);

                l = alpha * l + beta;
                m = m_new;

                const int64_t pf_t = t + 2;
                if (pf_t < n_kv) {
                    const size_t pf_off = kv_token_head_offset(kv_view, stream, pf_t, kv_head);
                    const char * base = (const char *) (kv_view->base + pf_off);
                    _mm_prefetch(base, _MM_HINT_T0);
                    _mm_prefetch(base + head_dim, _MM_HINT_T0);
                    const size_t scale_off = 2 * (size_t) head_dim;
                    _mm_prefetch(base + scale_off, _MM_HINT_T0);
                    _mm_prefetch(base + scale_off + sizeof(ggml_fp16_t), _MM_HINT_T0);
                }
            }
        }

        if (l > 0.0f) {
            const float inv_l = 1.0f / l;
            scale_row_f32_avx2(y_ptr, inv_l, (int) head_dim);
        }
    }
}
