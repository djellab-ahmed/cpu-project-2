#pragma once

#include "gptoss.h"
#include "common.h"

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;  // max drafted tokens
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft
};

struct common_speculative * common_speculative_init(
        struct gptoss_context * ctx_tgt,
        struct gptoss_context * ctx_dft
);

void common_speculative_free(struct common_speculative * spec);

bool common_speculative_are_compatible(
        const struct gptoss_context * ctx_tgt,
        const struct gptoss_context * ctx_dft);

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest);

// sample up to n_draft tokens and add them to the batch using the draft model
gptoss_tokens common_speculative_gen_draft(
               struct common_speculative * spec,
        struct common_speculative_params   params,
                      const gptoss_tokens & prompt,
                             gptoss_token   id_last);
