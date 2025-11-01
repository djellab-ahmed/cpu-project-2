#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "gptoss.h"

struct gptoss_model_deleter {
    void operator()(gptoss_model * model) { gptoss_model_free(model); }
};

struct gptoss_context_deleter {
    void operator()(gptoss_context * context) { gptoss_free(context); }
};

struct gptoss_sampler_deleter {
    void operator()(gptoss_sampler * sampler) { gptoss_sampler_free(sampler); }
};

struct gptoss_adapter_lora_deleter {
    void operator()(gptoss_adapter_lora * adapter) { gptoss_adapter_lora_free(adapter); }
};

typedef std::unique_ptr<gptoss_model, gptoss_model_deleter> gptoss_model_ptr;
typedef std::unique_ptr<gptoss_context, gptoss_context_deleter> gptoss_context_ptr;
typedef std::unique_ptr<gptoss_sampler, gptoss_sampler_deleter> gptoss_sampler_ptr;
typedef std::unique_ptr<gptoss_adapter_lora, gptoss_adapter_lora_deleter> gptoss_adapter_lora_ptr;
