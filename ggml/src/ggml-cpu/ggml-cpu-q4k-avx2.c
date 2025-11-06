#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "ops.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define GGML_TLS _Thread_local
#elif defined(_MSC_VER)
#define GGML_TLS __declspec(thread)
#else
#define GGML_TLS __thread
#endif

static GGML_TLS float * tls_decode_x = NULL;
static GGML_TLS size_t  tls_decode_x_cap = 0;

static inline int qgemv_dbg(void) {
    const char * e = getenv("GPTOSS_QGEMV_DEBUG");
    return e != NULL && e[0] != '\0' && e[0] != '0';
}

static int q4k_logged;

static inline void * ggml_qgemv_tls_realloc(void ** ptr, size_t * cap, size_t need, size_t elem_sz) {
    if (need == 0) {
        return *ptr;
    }

    if (*cap < need) {
        if (*ptr != NULL) {
            ggml_aligned_free(*ptr, (*cap) * elem_sz);
        }
        void * const new_ptr = ggml_aligned_malloc(need * elem_sz);
        GGML_ASSERT(new_ptr != NULL);
        *ptr = new_ptr;
        *cap = need;
    }

    return *ptr;
}

static inline const char * ggml_qgemv_row_ptr_from_index(
        const char * base,
        int64_t row_index,
        int64_t rows_per_i2,
        int64_t rows_per_i3,
        int64_t tiles_i2,
        int64_t tiles_i3,
        size_t nb01,
        size_t nb02,
        size_t nb03) {

    int64_t tmp = row_index;
    const int64_t i3 = tiles_i3 > 0 ? tmp / rows_per_i3 : 0;
    tmp -= i3 * rows_per_i3;
    const int64_t i2 = tiles_i2 > 0 ? tmp / rows_per_i2 : 0;
    const int64_t i1 = tmp - i2 * rows_per_i2;

    return base + i1 * nb01 + i2 * nb02 + i3 * nb03;
}

static inline void ggml_qgemv_get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

#if defined(__AVX2__)

#include <immintrin.h>

static inline void ggml_qgemv_accumulate_low32(
        const uint8_t * qbytes,
        const float * xptr,
        const __m256 scale,
        const __m256 min,
        __m256 * acc) {

    const __m128i mask0f = _mm_set1_epi16(0x000F);
    const __m128i zero = _mm_setzero_si128();

    for (int off = 0; off < 32; off += 8) {
        const __m128i raw = _mm_loadl_epi64((const __m128i *)(qbytes + off));
        const __m128i raw16 = _mm_unpacklo_epi8(raw, zero);
        const __m128i nib16 = _mm_and_si128(raw16, mask0f);
        const __m256i nib32 = _mm256_cvtepu16_epi32(nib16);
        const __m256 nibf = _mm256_cvtepi32_ps(nib32);
        const __m256 scaled = _mm256_mul_ps(nibf, scale);
        const __m256 values = _mm256_sub_ps(scaled, min);
        const __m256 x = _mm256_loadu_ps(xptr + off);
        const __m256 prod = _mm256_mul_ps(values, x);
        *acc = _mm256_add_ps(*acc, prod);
    }
}

static inline void ggml_qgemv_accumulate_high32(
        const uint8_t * qbytes,
        const float * xptr,
        const __m256 scale,
        const __m256 min,
        __m256 * acc) {

    const __m128i mask0f = _mm_set1_epi16(0x000F);
    const __m128i zero = _mm_setzero_si128();

    for (int off = 0; off < 32; off += 8) {
        const __m128i raw = _mm_loadl_epi64((const __m128i *)(qbytes + off));
        const __m128i raw16 = _mm_unpacklo_epi8(raw, zero);
        const __m128i hibits = _mm_srli_epi16(raw16, 4);
        const __m128i nib16 = _mm_and_si128(hibits, mask0f);
        const __m256i nib32 = _mm256_cvtepu16_epi32(nib16);
        const __m256 nibf = _mm256_cvtepi32_ps(nib32);
        const __m256 scaled = _mm256_mul_ps(nibf, scale);
        const __m256 values = _mm256_sub_ps(scaled, min);
        const __m256 x = _mm256_loadu_ps(xptr + off);
        const __m256 prod = _mm256_mul_ps(values, x);
        *acc = _mm256_add_ps(*acc, prod);
    }
}

void ggml_mul_mat_q4k_decode_avx2(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const struct ggml_tensor * w,
        const struct ggml_tensor * x) {

    const struct ggml_tensor * const w_tensor = w;
    const struct ggml_tensor * const x_tensor = x;

    GGML_ASSERT(ggml_cpu_is_q4k_family(w_tensor->type));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x_tensor->ne[1] == 1);
    GGML_ASSERT(
        x_tensor->type == GGML_TYPE_F32 ||
        x_tensor->type == GGML_TYPE_F16 ||
        x_tensor->type == GGML_TYPE_BF16 ||
        x_tensor->type == GGML_TYPE_Q8_K);

    if (!q4k_logged && qgemv_dbg() && params->ith == 0) {
        q4k_logged = 1;
        fprintf(stderr, "[qgemv] Q4_K AVX2 decode kernel active (n=1)\n");
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t cols = w_tensor->ne[0];
    GGML_ASSERT(cols % QK_K == 0);

    const int64_t rows_per_mat = w_tensor->ne[1];
    const int64_t tiles_i2 = w_tensor->ne[2] > 0 ? w_tensor->ne[2] : 1;
    const int64_t tiles_i3 = w_tensor->ne[3] > 0 ? w_tensor->ne[3] : 1;

    const int64_t total_rows = rows_per_mat * tiles_i2 * tiles_i3;
    const int64_t r0 = total_rows * ith / nth;
    const int64_t r1 = total_rows * (ith + 1) / nth;
    if (r0 >= r1 || total_rows == 0) {
        return;
    }

    GGML_ASSERT(w_tensor->nb[0] == ggml_type_size(w_tensor->type));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const size_t nb01 = w_tensor->nb[1];
    const size_t nb02 = w_tensor->nb[2];
    const size_t nb03 = w_tensor->nb[3];

    const size_t nb10 = x_tensor->nb[0];
    GGML_ASSERT(nb10 == ggml_type_size(x_tensor->type));
    const size_t nb12 = x_tensor->nb[2];
    const size_t nb13 = x_tensor->nb[3];

    const size_t nb0 = dst->nb[0];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const int64_t rows_per_i2 = rows_per_mat;
    const int64_t rows_per_i3 = rows_per_mat * tiles_i2;

    float * x_f32_tmp = NULL;
    if (x_tensor->type != GGML_TYPE_F32) {
        x_f32_tmp = ggml_qgemv_tls_realloc((void **) &tls_decode_x, &tls_decode_x_cap, (size_t) cols, sizeof(float));
    }

    const char * GGML_RESTRICT x_base = (const char *) x_tensor->data;
    const char * GGML_RESTRICT w_base = (const char *) w_tensor->data;
    char * GGML_RESTRICT dst_base = (char *) dst->data;

    int64_t prev_i2 = -1;
    int64_t prev_i3 = -1;
    const float * x_f32_cur = NULL;

    for (int64_t row_index = r0; row_index < r1; ++row_index) {
        int64_t tmp = row_index;
        const int64_t i3 = tiles_i3 > 0 ? tmp / rows_per_i3 : 0;
        tmp -= i3 * rows_per_i3;
        const int64_t i2 = tiles_i2 > 0 ? tmp / rows_per_i2 : 0;
        const int64_t i1 = tmp - i2 * rows_per_i2;

        if (i2 != prev_i2 || i3 != prev_i3) {
            prev_i2 = i2;
            prev_i3 = i3;

            const char * x_ptr = x_base + i2 * nb12 + i3 * nb13;

            if (x_tensor->type == GGML_TYPE_F32) {
                x_f32_cur = (const float *) x_ptr;
            } else if (x_tensor->type == GGML_TYPE_F16) {
                GGML_ASSERT(x_f32_tmp != NULL);
                ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) x_ptr, x_f32_tmp, cols);
                x_f32_cur = x_f32_tmp;
            } else if (x_tensor->type == GGML_TYPE_BF16) {
                GGML_ASSERT(x_f32_tmp != NULL);
                ggml_cpu_bf16_to_fp32((const ggml_bf16_t *) x_ptr, x_f32_tmp, cols);
                x_f32_cur = x_f32_tmp;
            } else if (x_tensor->type == GGML_TYPE_Q8_K) {
                GGML_ASSERT(x_f32_tmp != NULL);
                dequantize_row_q8_K((const block_q8_K *) x_ptr, x_f32_tmp, cols);
                x_f32_cur = x_f32_tmp;
            } else {
                GGML_ABORT("Q4_K decode kernel: unsupported activation type");
            }

            _mm_prefetch((const char *) x_f32_cur, _MM_HINT_T0);
        }

        const char * w_row_ptr = w_base + i1 * nb01 + i2 * nb02 + i3 * nb03;
        const block_q4_K * w_row = (const block_q4_K *) w_row_ptr;

        __m256 acc = _mm256_setzero_ps();

        const int64_t n_blocks = cols / QK_K;
        for (int64_t blk = 0; blk < n_blocks; ++blk) {
            const block_q4_K * w_block = w_row + blk;
            const uint8_t * qbytes = w_block->qs;

            const float d = GGML_FP16_TO_FP32(w_block->d);
            const float dmin = GGML_FP16_TO_FP32(w_block->dmin);

            uint8_t sc, m;
            int is = 0;

            for (int chunk = 0; chunk < QK_K; chunk += 64) {
                ggml_qgemv_get_scale_min_k4(is + 0, w_block->scales, &sc, &m);
                const __m256 scale0 = _mm256_set1_ps(d * sc);
                const __m256 min0 = _mm256_set1_ps(dmin * m);
                ggml_qgemv_accumulate_low32(qbytes, x_f32_cur + blk * QK_K + chunk, scale0, min0, &acc);

                ggml_qgemv_get_scale_min_k4(is + 1, w_block->scales, &sc, &m);
                const __m256 scale1 = _mm256_set1_ps(d * sc);
                const __m256 min1 = _mm256_set1_ps(dmin * m);
                ggml_qgemv_accumulate_high32(qbytes, x_f32_cur + blk * QK_K + chunk + 32, scale1, min1, &acc);

                qbytes += 32;
                is += 2;
            }

            if (blk + 1 < n_blocks) {
                _mm_prefetch(((const char *) (w_row + blk + 1)) + 256, _MM_HINT_T0);
            }
        }

        float acc_buf[8];
        _mm256_storeu_ps(acc_buf, acc);
        float acc_scalar = acc_buf[0] + acc_buf[1] + acc_buf[2] + acc_buf[3] + acc_buf[4] + acc_buf[5] + acc_buf[6] + acc_buf[7];

        float * dst_ptr = (float *)(dst_base + i1 * nb0 + i2 * nb2 + i3 * nb3);
        *dst_ptr = acc_scalar;

        if (row_index + 1 < r1) {
            const char * next_row = ggml_qgemv_row_ptr_from_index(
                w_base,
                row_index + 1,
                rows_per_i2,
                rows_per_i3,
                tiles_i2,
                tiles_i3,
                nb01,
                nb02,
                nb03);
            _mm_prefetch(next_row, _MM_HINT_T0);
        }
    }
}

#else // !__AVX2__

void ggml_mul_mat_q4k_decode_avx2(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const struct ggml_tensor * w,
        const struct ggml_tensor * x) {
    (void) params;
    (void) dst;
    (void) w;
    (void) x;
    GGML_ABORT("Q4_K decode AVX2 kernel requires AVX2 support");
}

#endif // __AVX2__

