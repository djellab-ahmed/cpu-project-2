#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
enum gptoss_kv_dtype : uint8_t {
#else
typedef uint8_t gptoss_kv_dtype;
enum {
#endif
    GPTOSS_KV_F16      = 0,
    GPTOSS_KV_Q8_ROWROW = 1,
#ifdef __cplusplus
};
#else
};
#endif

struct gptoss_kv_view {
    gptoss_kv_dtype dtype_k;
    gptoss_kv_dtype dtype_v;
    // Interleaved per-(token,head) block: [K_i8[hd]][V_i8[hd]][sK_fp16][sV_fp16]
    uint8_t * base;
    size_t    stride_token_bytes;
    size_t    stride_head_bytes;
    size_t    stride_stream_bytes;
    size_t    block_bytes;
    int       head_dim;
    int       n_heads_kv;
    uint8_t * scales_k;   // fp16 storage; row,row scheme
    uint8_t * scales_v;   // fp16 storage; row,row scheme
    int       tile_tokens; // e.g., 128
    int       interleaved;
#ifdef __linux__
    int       huge_pages;
#endif
};

