// SPDX-License-Identifier: MIT
#include "gptoss-kv-layout.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <cstdio>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

static bool env_flag(const char *name, bool defv) {
    const char *v = std::getenv(name);
    if (!v || !*v) {
        return defv;
    }
    return *v != '0';
}

static int env_int(const char *name, int defv) {
    const char *v = std::getenv(name);
    if (!v || !*v) {
        return defv;
    }
    return std::atoi(v);
}

static size_t round_up(size_t v, size_t align) {
    if (align == 0) {
        return v;
    }
    const size_t mask = align - 1;
    return (v + mask) & ~mask;
}

#if defined(__linux__)
static constexpr size_t HUGEPAGE_2M = 2ull * 1024ull * 1024ull;
#endif

} // namespace

int32_t gptoss_kv_default_tile(int64_t /*head_dim*/) {
    int tile = env_int("GPTOSS_KV_TILE", 0);
    if (tile <= 0) {
        tile = 256;
    }
    tile = std::clamp(tile, 32, 2048);
    return tile;
}

int32_t gptoss_kv_pad_tokens() {
    int pad = env_int("GPTOSS_KV_PAD", 64);
    pad = std::clamp(pad, 1, 2048);
    return pad;
}

bool gptoss_kv_interleave_enabled() {
    return env_flag("GPTOSS_KV_INTERLEAVE", true);
}

bool gptoss_kv_prefer_hugepages() {
    return env_flag("GPTOSS_KV_HUGEPAGES", true);
}

static std::atomic<int> g_kv_debug_once{0};

void gptoss_kv_debug_once(const gptoss_kv_view &v, const char *tag) {
    if (!env_flag("GPTOSS_KV_DEBUG", false)) {
        return;
    }
    if (g_kv_debug_once.fetch_add(1) == 0) {
        fprintf(stderr,
                "[kv-layout] %s interleaved=%d hugepages=%d dtype=%s D=%lld heads=%lld streams=%lld cap=%lld stride_token=%zu stride_head=%zu tile=%d pad=%d\n",
                tag ? tag : "init",
                v.interleaved ? 1 : 0,
                v.hugepages ? 1 : 0,
                ggml_type_name(v.dtype),
                (long long) v.head_dim,
                (long long) v.n_head_kv,
                (long long) v.n_stream,
                (long long) v.cap_tokens,
                v.stride_token_bytes,
                v.stride_head_bytes,
                v.tile_tok,
                v.pad_tok);
    }
}

static void * alloc_buffer(size_t size, bool prefer_huge, bool &huge_used, bool &via_mmap, size_t &mapped_bytes) {
    huge_used = false;
    via_mmap = false;
    mapped_bytes = size;
#if defined(__linux__)
    if (prefer_huge) {
#if defined(MAP_HUGETLB)
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_SHIFT 26
#define MAP_HUGE_2MB   (21 << MAP_HUGE_SHIFT)
#endif
        mapped_bytes = round_up(size, HUGEPAGE_2M);
        void *ptr = mmap(nullptr, mapped_bytes,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
#ifdef MAP_POPULATE
                         | MAP_POPULATE
#endif
                         | MAP_HUGE_2MB,
                         -1, 0);
        if (ptr != MAP_FAILED) {
            huge_used = true;
            via_mmap = true;
            std::memset(ptr, 0, mapped_bytes);
            return ptr;
        }
#endif
        mapped_bytes = size;
        void *ptr_fallback = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr_fallback != MAP_FAILED) {
#if defined(MADV_HUGEPAGE)
            madvise(ptr_fallback, size, MADV_HUGEPAGE);
#endif
            via_mmap = true;
            std::memset(ptr_fallback, 0, size);
            return ptr_fallback;
        }
    }
#endif
    void *buf = nullptr;
    if (posix_memalign(&buf, 64u, size) != 0) {
        return nullptr;
    }
    std::memset(buf, 0, size);
    return buf;
}

bool gptoss_kv_alloc(gptoss_kv_view &view,
                     ggml_type dtype,
                     int64_t head_dim,
                     int64_t n_head_kv,
                     int64_t cap_tokens,
                     int64_t n_stream,
                     bool    prefer_hugepages,
                     int32_t tile_tok,
                     int32_t pad_tok) {
    if (cap_tokens <= 0 || n_stream <= 0 || n_head_kv <= 0 || head_dim <= 0) {
        return false;
    }

    gptoss_kv_free(view);

    view.dtype   = dtype;
    view.elem    = ggml_type_size(dtype);
    view.head_dim = head_dim;
    view.n_head_kv = n_head_kv;
    view.n_stream  = n_stream;
    view.tile_tok  = tile_tok > 0 ? tile_tok : gptoss_kv_default_tile(head_dim);
    view.pad_tok   = pad_tok  > 0 ? pad_tok  : gptoss_kv_pad_tokens();

    if (view.pad_tok > 1 && (cap_tokens % view.pad_tok) != 0) {
        cap_tokens += view.pad_tok - (cap_tokens % view.pad_tok);
    }

    view.cap_tokens = cap_tokens;
    view.used_tokens = 0;

    const size_t block_bytes = (size_t) head_dim * view.elem;
    view.stride_head_bytes   = block_bytes * 2u;
    view.stride_token_bytes  = view.stride_head_bytes * (size_t) n_head_kv;
    view.stride_stream_bytes = view.stride_token_bytes * (size_t) cap_tokens;

    const size_t total = view.stride_stream_bytes * (size_t) n_stream;

    bool huge_used = false;
    bool via_mmap = false;
    size_t mapped_bytes = total;
    void *buffer = alloc_buffer(total, prefer_hugepages, huge_used, via_mmap, mapped_bytes);
    if (!buffer) {
        return false;
    }

    view.base         = buffer;
    view.bytes        = total;
    view.mapped_bytes = mapped_bytes;
    view.hugepages    = huge_used;
    view.via_mmap     = via_mmap;
    view.interleaved  = true;

    gptoss_kv_debug_once(view, "alloc");
    return true;
}

void gptoss_kv_free(gptoss_kv_view &view) {
    if (!view.base) {
        return;
    }
#if defined(__linux__)
    if (view.via_mmap) {
        munmap(view.base, view.mapped_bytes ? view.mapped_bytes : view.bytes);
    } else {
        free(view.base);
    }
#else
    free(view.base);
#endif
    view = gptoss_kv_view{};
}

void gptoss_kv_store(gptoss_kv_view &view,
                     int64_t stream_idx,
                     int64_t token_idx,
                     int64_t head_idx,
                     const void *srcK,
                     const void *srcV) {
    GGML_ASSERT(view.base != nullptr);
    GGML_ASSERT(stream_idx >= 0 && stream_idx < view.n_stream);
    GGML_ASSERT(token_idx >= 0 && token_idx < view.cap_tokens);
    GGML_ASSERT(head_idx >= 0 && head_idx < view.n_head_kv);

    const size_t block_bytes = (size_t) view.head_dim * view.elem;
    uint8_t *dst = static_cast<uint8_t *>(view.base)
                 + (size_t) stream_idx * view.stride_stream_bytes
                 + (size_t) token_idx  * view.stride_token_bytes
                 + (size_t) head_idx   * view.stride_head_bytes;

    std::memcpy(dst, srcK, block_bytes);
    std::memcpy(dst + block_bytes, srcV, block_bytes);
    if (token_idx + 1 > view.used_tokens) {
        view.used_tokens = token_idx + 1;
    }
}

struct ggml_tensor * gptoss_make_k_view(struct ggml_context *ctx,
                                        const gptoss_kv_view &view,
                                        int64_t n_kv,
                                        int64_t n_stream) {
    int64_t ne[4] = { view.head_dim, view.n_head_kv, n_kv, n_stream };
    struct ggml_tensor *t = ggml_new_tensor(ctx, view.dtype, 4, ne);
    t->data = static_cast<uint8_t *>(view.base);
    t->nb[0] = (int64_t) view.elem;
    t->nb[1] = (int64_t) view.stride_head_bytes;
    t->nb[2] = (int64_t) view.stride_token_bytes;
    t->nb[3] = (int64_t) view.stride_stream_bytes;
    return t;
}

struct ggml_tensor * gptoss_make_v_view(struct ggml_context *ctx,
                                        const gptoss_kv_view &view,
                                        int64_t n_kv,
                                        int64_t n_stream) {
    int64_t ne[4] = { view.head_dim, view.n_head_kv, n_kv, n_stream };
    struct ggml_tensor *t = ggml_new_tensor(ctx, view.dtype, 4, ne);
    const size_t block_bytes = (size_t) view.head_dim * view.elem;
    t->data = static_cast<uint8_t *>(view.base) + block_bytes;
    t->nb[0] = (int64_t) view.elem;
    t->nb[1] = (int64_t) view.stride_head_bytes;
    t->nb[2] = (int64_t) view.stride_token_bytes;
    t->nb[3] = (int64_t) view.stride_stream_bytes;
    return t;
}
