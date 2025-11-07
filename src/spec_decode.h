#pragma once

#include <random>
#include <string>
#include <vector>

#include "gptoss.h"

struct spec_cfg {
    int   max_propose  = 8;
    float min_accept   = 0.0f;
    bool  greedy_draft = true;
    bool  debug        = false;
};

struct spec_ctx {
    gptoss_model *   verifier_model = nullptr;
    gptoss_context * verifier_ctx   = nullptr;
    gptoss_model *   draft_model    = nullptr;
    gptoss_context * draft_ctx      = nullptr;

    std::vector<float> p_next;
    std::vector<float> q_next;
    std::vector<int>   candidate_order;
    std::vector<int>   candidate_scratch;
    std::vector<int>   proposals;
    std::vector<float> q_token_prob;
    std::vector<float> p_token_prob;

    std::vector<uint8_t> verifier_state;
    std::vector<uint8_t> draft_state;
};

struct spec_metrics {
    uint64_t steps     = 0;
    uint64_t proposed  = 0;
    uint64_t accepted  = 0;
    double   ewma_accept = 0.0;
};

int spec_step(
        spec_ctx & sx,
        const spec_cfg & cfg,
        std::vector<gptoss_token> & out_tokens,
        spec_metrics & mx,
        const std::vector<gptoss_token> & recent_for_repetition,
        float temperature,
        int   top_k,
        float top_p,
        float repeat_penalty,
        int   repeat_last_n,
        std::mt19937 & rng);

