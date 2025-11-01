#pragma once

// TODO: rename gptoss-sampling.h/.cpp to gptoss-sampler.h/.cpp ?

#include "gptoss.h"

#include <vector>

struct gptoss_vocab;
struct gptoss_grammar;

// sampler chain

struct gptoss_sampler_chain {
    gptoss_sampler_chain_params params;

    std::vector<struct gptoss_sampler *> samplers;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct gptoss_sampler * gptoss_sampler_init_dry_testing(
                         int32_t   context_size,
                           float   dry_multiplier,
                           float   dry_base,
                         int32_t   dry_allowed_length,
                         int32_t   dry_penalty_last_n,
  const std::vector<std::vector<gptoss_token>>& seq_breakers);
