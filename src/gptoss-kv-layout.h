// SPDX-License-Identifier: MIT
#pragma once
#include <stddef.h>
#include <stdint.h>

// Token-major, K/V-interleaved KV cache view.
// Per token t in a stream:
//   [ K(head 0), V(head 0), K(head 1), V(head 1), ..., K(head n-1), V(head n-1) ]
// Each K or V block is a contiguous vector of length head_dim elements.
struct gptoss_kv_view {
    uint8_t  *base;          // base pointer (shared by both K & V views)
    size_t    stride_head;   // bytes to step head within a token (== 2 * block_bytes)
    size_t    stride_token;  // bytes to step token (== n_head_kv * stride_head)
    size_t    stride_stream; // bytes to step stream
    size_t    block_bytes;   // bytes for one contiguous K (or V) block (== head_dim * dtype_size)
    int64_t   head_dim;      // elements per K/V block
    int       n_head_kv;     // KV heads
    int       tile_pad;      // token tile pad (e.g., 64/128)
    int       interleaved;   // 1 if K/V are interleaved as above
#ifdef __linux__
    int       huge_pages;    // 1 if mapped with MAP_HUGETLB
#endif
};

// Returns a sane default tile pad (power-of-two in [32, 2048]),
// honoring GPTOSS_KV_TILE if set.
int gptoss_kv_default_tile_pad(void);
