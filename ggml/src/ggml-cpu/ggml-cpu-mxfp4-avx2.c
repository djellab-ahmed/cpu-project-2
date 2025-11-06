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

static inline int qgemv_dbg(void) {
    const char * e = getenv("GPTOSS_QGEMV_DEBUG");
    return e != NULL && e[0] != '\0' && e[0] != '0';
}

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

#if defined(__AVX2__) && defined(GGML_TYPE_MXFP4)

static int mxfp4_logged;

static GGML_TLS float * tls_decode_x = NULL;
static GGML_TLS size_t  tls_decode_x_cap = 0;

#include <immintrin.h>

static inline void ggml_mxfp4_accumulate16(const __m128i values,
        const __m256 scale,
        const float * GGML_RESTRICT xptr,
        __m256 * acc) {
    const __m128i lo_i16 = _mm_cvtepi8_epi16(values);
    const __m128i hi_i16 = _mm_cvtepi8_epi16(_mm_srli_si128(values, 8));

    const __m256i lo_i32 = _mm256_cvtepi16_epi32(lo_i16);
    const __m256i hi_i32 = _mm256_cvtepi16_epi32(hi_i16);

    __m256 vlo = _mm256_cvtepi32_ps(lo_i32);
    __m256 vhi = _mm256_cvtepi32_ps(hi_i32);

    vlo = _mm256_mul_ps(vlo, scale);
    vhi = _mm256_mul_ps(vhi, scale);

    const __m256 x0 = _mm256_loadu_ps(xptr + 0);
    const __m256 x1 = _mm256_loadu_ps(xptr + 8);

#if defined(__FMA__)
    *acc = _mm256_fmadd_ps(vlo, x0, *acc);
    *acc = _mm256_fmadd_ps(vhi, x1, *acc);
#else
    *acc = _mm256_add_ps(*acc, _mm256_mul_ps(vlo, x0));
    *acc = _mm256_add_ps(*acc, _mm256_mul_ps(vhi, x1));
#endif
}

void ggml_mul_mat_mxfp4_decode_avx2(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const struct ggml_tensor * w,
        const struct ggml_tensor * x) {

    const struct ggml_tensor * const w_tensor = w;
    const struct ggml_tensor * const x_tensor = x;

    GGML_ASSERT(w_tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x_tensor->ne[1] == 1);
    GGML_ASSERT(
        x_tensor->type == GGML_TYPE_F32 ||
        x_tensor->type == GGML_TYPE_F16 ||
        x_tensor->type == GGML_TYPE_BF16);

    if (!mxfp4_logged && qgemv_dbg() && params->ith == 0) {
        mxfp4_logged = 1;
        fprintf(stderr, "[qgemv] MXFP4 AVX2 decode kernel active (n=1)\n");
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t cols = w_tensor->ne[0];
    GGML_ASSERT(cols % QK_MXFP4 == 0);

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

    // kvalues_mxfp4 holds the canonical 16-entry E2M1 LUT shared with the quantizers.
    const __m128i lut = _mm_loadu_si128((const __m128i *) kvalues_mxfp4);
    const __m128i mask0f = _mm_set1_epi8(0x0F);

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
            } else {
                GGML_ABORT("MXFP4 decode kernel: unsupported activation type");
            }

            _mm_prefetch((const char *) x_f32_cur, _MM_HINT_T0);
        }

        const char * w_row_ptr = w_base + i1 * nb01 + i2 * nb02 + i3 * nb03;
        const block_mxfp4 * w_row = (const block_mxfp4 *) w_row_ptr;

        GGML_ASSERT(x_f32_cur != NULL);

        __m256 acc = _mm256_setzero_ps();

        const int64_t n_blocks = cols / QK_MXFP4;
        for (int64_t blk = 0; blk < n_blocks; ++blk) {
            const block_mxfp4 * w_block = w_row + blk;
            const __m256 scale = _mm256_set1_ps(GGML_E8M0_TO_FP32_HALF(w_block->e));

            const __m128i packed = _mm_loadu_si128((const __m128i *) w_block->qs);
            const __m128i lo_nibbles = _mm_and_si128(packed, mask0f);
            const __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), mask0f);

            const __m128i lo_vals = _mm_shuffle_epi8(lut, lo_nibbles);
            const __m128i hi_vals = _mm_shuffle_epi8(lut, hi_nibbles);

            ggml_mxfp4_accumulate16(lo_vals, scale, x_f32_cur + blk * QK_MXFP4 + 0, &acc);
            ggml_mxfp4_accumulate16(hi_vals, scale, x_f32_cur + blk * QK_MXFP4 + 16, &acc);

            if (blk + 1 < n_blocks) {
                _mm_prefetch(((const char *) (w_row + blk + 1)) + 256, _MM_HINT_T0);
            }
        }

        __m128 acc_low = _mm256_castps256_ps128(acc);
        __m128 acc_high = _mm256_extractf128_ps(acc, 1);
        __m128 acc_sum = _mm_add_ps(acc_low, acc_high);
        acc_sum = _mm_hadd_ps(acc_sum, acc_sum);
        acc_sum = _mm_hadd_ps(acc_sum, acc_sum);
        const float acc_scalar = _mm_cvtss_f32(acc_sum);

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

#else

void ggml_mul_mat_mxfp4_decode_avx2(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const struct ggml_tensor * w,
        const struct ggml_tensor * x) {
    (void) params;
    (void) dst;
    (void) w;
    (void) x;
    GGML_ABORT("MXFP4 decode AVX2 kernel requires AVX2 support and MXFP4 type");
}

#endif
