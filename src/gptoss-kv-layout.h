// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>

#include "ggml.h"
#include "ggml-backend.h"

struct gptoss_kv_view {
    void * base         = nullptr;
    size_t bytes        = 0;
    size_t mapped_bytes = 0;
    size_t align        = 64;

    ggml_type dtype     = GGML_TYPE_F16;
    size_t elem         = 2;
    int64_t head_dim    = 0;
    int64_t n_head_kv   = 0;
    int64_t n_stream    = 1;

    size_t stride_head_bytes   = 0;
    size_t stride_token_bytes  = 0;
    size_t stride_stream_bytes = 0;

    int64_t cap_tokens  = 0;
    int64_t used_tokens = 0;

    int32_t tile_tok    = 0;
    int32_t pad_tok     = 64;

    bool interleaved    = true;
    bool hugepages      = false;
    bool via_mmap       = false;

    ggml_backend_buffer_t buffer = nullptr;
};

int32_t gptoss_kv_default_tile(int64_t head_dim);
int32_t gptoss_kv_pad_tokens();
bool    gptoss_kv_interleave_enabled();
bool    gptoss_kv_prefer_hugepages();

bool gptoss_kv_alloc(gptoss_kv_view &view,
                     ggml_type dtype,
                     int64_t head_dim,
                     int64_t n_head_kv,
                     int64_t cap_tokens,
                     int64_t n_stream,
                     bool    prefer_hugepages,
                     int32_t tile_tok,
                     int32_t pad_tok);

void gptoss_kv_free(gptoss_kv_view &view);

void gptoss_kv_store(gptoss_kv_view &view,
                     int64_t stream_idx,
                     int64_t token_idx,
                     int64_t head_idx,
                     const void *srcK,
                     const void *srcV);

struct ggml_tensor * gptoss_make_k_view(struct ggml_context *ctx,
                                        const gptoss_kv_view &view,
                                        int64_t n_kv,
                                        int64_t n_stream);

struct ggml_tensor * gptoss_make_v_view(struct ggml_context *ctx,
                                        const gptoss_kv_view &view,
                                        int64_t n_kv,
                                        int64_t n_stream);

void gptoss_kv_debug_once(const gptoss_kv_view &view, const char *tag);
