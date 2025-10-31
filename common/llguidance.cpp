#include "sampling.h"
#include "log.h"

#ifdef GPTOSS_USE_LLGUIDANCE

#    include "llguidance.h"
#    include <cmath>

struct gptoss_sampler_llg {
    const gptoss_vocab * vocab;
    std::string         grammar_kind;
    std::string         grammar_data;
    LlgTokenizer *      tokenizer;
    LlgMatcher *        grammar;
};

static LlgMatcher * gptoss_sampler_llg_new(LlgTokenizer * tokenizer, const char * grammar_kind,
                                          const char * grammar_data) {
    LlgConstraintInit cinit;
    llg_constraint_init_set_defaults(&cinit, tokenizer);
    const char * log_level = getenv("LLGUIDANCE_LOG_LEVEL");
    if (log_level && *log_level) {
        cinit.log_stderr_level = atoi(log_level);
    }
    auto c = llg_new_matcher(&cinit, grammar_kind, grammar_data);
    if (llg_matcher_get_error(c)) {
        LOG_ERR("llg error: %s\n", llg_matcher_get_error(c));
        llg_free_matcher(c);
        return nullptr;
    }

    return c;
}

static const char * gptoss_sampler_llg_name(const gptoss_sampler * /*smpl*/) {
    return "llguidance";
}

static void gptoss_sampler_llg_accept_impl(gptoss_sampler * smpl, gptoss_token token) {
    auto * ctx = (gptoss_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        llg_matcher_consume_token(ctx->grammar, token);
    }
}

static void gptoss_sampler_llg_apply(gptoss_sampler * smpl, gptoss_token_data_array * cur_p) {
    auto * ctx = (gptoss_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        const uint32_t * mask = llg_matcher_get_mask(ctx->grammar);
        if (mask == nullptr) {
            if (llg_matcher_compute_mask(ctx->grammar) == 0) {
                mask = llg_matcher_get_mask(ctx->grammar);
            } else {
                LOG_ERR("llg error: %s\n", llg_matcher_get_error(ctx->grammar));
                llg_free_matcher(ctx->grammar);
                ctx->grammar = nullptr;
                return;
            }
        }

        for (size_t i = 0; i < cur_p->size; ++i) {
            auto token = cur_p->data[i].id;
            if ((mask[token / 32] & (1 << (token % 32))) == 0) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    }
}

static void gptoss_sampler_llg_reset(gptoss_sampler * smpl) {
    auto * ctx = (gptoss_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        llg_matcher_reset(ctx->grammar);
    }
}

static gptoss_sampler * gptoss_sampler_llg_clone(const gptoss_sampler * smpl) {
    const auto * ctx = (const gptoss_sampler_llg *) smpl->ctx;

    auto * result = gptoss_sampler_init_llg(ctx->vocab, nullptr, nullptr);

    // copy the state
    {
        auto * result_ctx = (gptoss_sampler_llg *) result->ctx;

        if (ctx->grammar) {
            result_ctx->grammar_kind = ctx->grammar_kind;
            result_ctx->grammar_data = ctx->grammar_data;
            result_ctx->grammar      = llg_clone_matcher(ctx->grammar);
            result_ctx->tokenizer    = llg_clone_tokenizer(ctx->tokenizer);
        }
    }

    return result;
}

static void gptoss_sampler_llg_free(gptoss_sampler * smpl) {
    const auto * ctx = (gptoss_sampler_llg *) smpl->ctx;

    if (ctx->grammar) {
        llg_free_matcher(ctx->grammar);
        llg_free_tokenizer(ctx->tokenizer);
    }

    delete ctx;
}

static gptoss_sampler_i gptoss_sampler_llg_i = {
    /* .name   = */ gptoss_sampler_llg_name,
    /* .accept = */ gptoss_sampler_llg_accept_impl,
    /* .apply  = */ gptoss_sampler_llg_apply,
    /* .reset  = */ gptoss_sampler_llg_reset,
    /* .clone  = */ gptoss_sampler_llg_clone,
    /* .free   = */ gptoss_sampler_llg_free,
};

static size_t gptoss_sampler_llg_tokenize_fn(const void * user_data, const uint8_t * bytes, size_t bytes_len,
                                            uint32_t * output_tokens, size_t output_tokens_len) {
    const gptoss_vocab * vocab = (const gptoss_vocab *) user_data;
    int                 r     = 0;
    try {
        r = gptoss_tokenize(vocab, (const char *) bytes, bytes_len, (int32_t *) output_tokens, output_tokens_len, false,
                           true);
    } catch (const std::exception & e) {
        GGML_ABORT("gptoss_tokenize failed: %s\n", e.what());
    }
    if (r < 0) {
        return -r;
    }
    return r;
}

static LlgTokenizer * gptoss_sampler_llg_new_tokenizer(const gptoss_vocab * vocab) {
    // TODO store the tokenizer in the vocab somehow
    static const gptoss_vocab * vocab_cache;
    static LlgTokenizer *      tokenizer_cache;

    if (vocab_cache == vocab) {
        return llg_clone_tokenizer(tokenizer_cache);
    }

    auto tok_eos = gptoss_vocab_eot(vocab);
    if (tok_eos == GPTOSS_TOKEN_NULL) {
        tok_eos = gptoss_vocab_eos(vocab);
    }

    size_t vocab_size = gptoss_vocab_n_tokens(vocab);

    auto token_lens       = new uint32_t[vocab_size];
    // we typically have ~7 bytes per token; let's go on the safe side here
    auto token_bytes_size = vocab_size * 16 + 1024 * 1024;
    auto token_bytes      = new uint8_t[token_bytes_size];

    size_t offset = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        size_t max_token = 1024;
        if (token_bytes_size - offset < max_token) {
            GGML_ABORT("token_bytes buffer too small\n");
        }

        gptoss_token token = i;
        auto        dp    = (char *) token_bytes + offset;
        auto        size  = gptoss_detokenize(vocab, &token, 1, dp, max_token, false, false);
        if (size < 0) {
            GGML_ABORT("gptoss_detokenize failed\n");
        }
        if (size == 0) {
            size = gptoss_detokenize(vocab, &token, 1, dp + 1, max_token - 1, false, true);
            if (size < 0) {
                GGML_ABORT("gptoss_detokenize failed\n");
            }
            if (size != 0) {
                *dp = '\xff';  // special token prefix marker
                size += 1;
            }
        }

        token_lens[i] = size;
        offset += size;
    }

    LlgTokenizerInit tinit = {
        /* .vocab_size                         = */ (uint32_t) vocab_size,
        /* .tok_eos                            = */ (uint32_t) tok_eos,
        /* .token_lens                         = */ token_lens,
        /* .token_bytes                        = */ token_bytes,
        /* .tokenizer_json                     = */ nullptr,
        /* .tokenize_assumes_string            = */ true,
        /* .tokenize_fn                        = */ gptoss_sampler_llg_tokenize_fn,
        /* .use_approximate_greedy_tokenize_fn = */ false,
        /* .tokenize_user_data                 = */ vocab,
        /* .slices                             = */ nullptr,
    };

    char           error_buffer[1024];
    LlgTokenizer * tokenizer = llg_new_tokenizer(&tinit, error_buffer, sizeof(error_buffer));

    delete[] token_bytes;
    delete[] token_lens;

    if (tokenizer == nullptr) {
        LOG_ERR("llg tokenizer error: %s\n", error_buffer);
        return tokenizer;
    }

    if (tokenizer_cache) {
        llg_free_tokenizer(tokenizer_cache);
    }
    vocab_cache     = vocab;
    tokenizer_cache = tokenizer;

    return llg_clone_tokenizer(tokenizer_cache);
}

gptoss_sampler * gptoss_sampler_init_llg(const gptoss_vocab * vocab, const char * grammar_kind,
                                       const char * grammar_data) {
    auto * ctx = new gptoss_sampler_llg;

    if (grammar_kind != nullptr && grammar_kind[0] != '\0') {
        auto tokenizer = gptoss_sampler_llg_new_tokenizer(vocab);
        *ctx           = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ grammar_kind,
            /* .grammar_data = */ grammar_data,
            /* .tokenizer    = */ tokenizer,
            /* .grammar      = */ gptoss_sampler_llg_new(tokenizer, grammar_kind, grammar_data),
        };
        if (ctx->grammar) {
            GGML_ASSERT(((size_t) gptoss_vocab_n_tokens(vocab) + 31) / 32 * 4 ==
                        llg_matcher_get_mask_byte_size(ctx->grammar));
        }
    } else {
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ {},
            /* .grammar_data = */ {},
            /* .tokenizer    = */ nullptr,
            /* .grammar      = */ nullptr,
        };
    }

    return gptoss_sampler_init(
        /* .iface = */ &gptoss_sampler_llg_i,
        /* .ctx   = */ ctx);
}

#else

gptoss_sampler * gptoss_sampler_init_llg(const gptoss_vocab *, const char *, const char *) {
    LOG_WRN("llguidance (cmake -DGPTOSS_LLGUIDANCE=ON) is not enabled");
    return nullptr;
}

#endif  // GPTOSS_USE_LLGUIDANCE
