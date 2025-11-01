#pragma once

#include "ggml.h" // for ggml_log_level

#include <string>
#include <vector>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define GPTOSS_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define GPTOSS_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define GPTOSS_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

GPTOSS_ATTRIBUTE_FORMAT(2, 3)
void gptoss_log_internal        (ggml_log_level level, const char * format, ...);
void gptoss_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define GPTOSS_LOG(...)       gptoss_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GPTOSS_LOG_INFO(...)  gptoss_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GPTOSS_LOG_WARN(...)  gptoss_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GPTOSS_LOG_ERROR(...) gptoss_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GPTOSS_LOG_DEBUG(...) gptoss_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GPTOSS_LOG_CONT(...)  gptoss_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() { /* do nothing */ }
};

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to gptoss_format ?
GPTOSS_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string gptoss_format_tensor_shape(const std::vector<int64_t> & ne);
std::string gptoss_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define GPTOSS_TENSOR_NAME_FATTN "__fattn__"
