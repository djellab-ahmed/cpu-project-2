// SPDX-License-Identifier: MIT
#pragma once
#include <stddef.h>
#include <stdint.h>

// View over a token-major, K/V-interleaved KV cache.
// Memory layout per token t:
//   [ K(head 0), V(head 0), K(head 1), V(head 1), ..., K(head n-1), V(head n-1) ]
// Each K/V block is a contiguous vector of head_dim elements in the chosen dtype.
struct gptoss_kv_view {
    uint8_t  *base;          // base pointer of underlying allocation (shared by K & V views)
    size_t    stride_head;   // bytes to step head within the same token (== 2 * block_bytes)
    size_t    stride_token;  // bytes to step token within the same stream (== n_head_kv * stride_head)
    size_t    stride_stream; // bytes to step stream
    size_t    block_bytes;   // bytes for one contiguous K (or V) block (== head_dim * dtype_size)
    int64_t   head_dim;      // elements per K/V block
    int       n_head_kv;     // # KV heads
    int       tile_pad;      // token tile pad (e.g. 64, 128)
    int       interleaved;   // 1 if the buffer is K/V interleaved (this design), 0 if legacy layout
#ifdef __linux__
    int       huge_pages;    // 1 if mapped by MAP_HUGETLB, 0 otherwise
#endif
};

// Returns default tile pad if env is not set or invalid.
int gptoss_kv_default_tile_pad(void);

