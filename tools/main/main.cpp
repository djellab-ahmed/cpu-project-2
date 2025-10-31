#include "gptoss.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct options {
    std::string model_path;
    std::string prompt;
    int32_t     threads        = -1;
    int32_t     threads_batch  = -1;
    uint32_t    context_size   = 0;
    uint32_t    batch_size     = 512;
    uint32_t    ubatch_size    = 512;
    int32_t     n_predict      = 256;
    bool        use_mlock      = false;
    ggml_numa_strategy numa    = GGML_NUMA_STRATEGY_DISABLED;
    int32_t     seed           = -1;
    float       temperature    = 0.8f;
    float       top_p          = 0.95f;
    int32_t     top_k          = 40;
    float       repeat_penalty = 1.1f;
    int32_t     repeat_last_n  = 64;
};

void print_usage(const char * program) {
    std::cerr << "Usage: " << program << " [options]\n"
              << "\nRequired:\n"
              << "  -m, --model PATH         Path to the GGUF model file\n"
              << "\nOptional:\n"
              << "  -p, --prompt TEXT        Text prompt to evaluate\n"
              << "  -n, --n-predict N        Number of tokens to generate (default 256)\n"
              << "  -t, --threads N          Threads for token generation\n"
              << "  -tb, --threads-batch N   Threads for prompt ingestion\n"
              << "      --ctx-size N         Override context window\n"
              << "      --ubatch-size N      Micro-batch size (default 512)\n"
              << "      --mlock              Request mlock for model weights\n"
              << "      --numa MODE          NUMA mode: none, distribute, isolate\n"
              << "      --seed N             RNG seed (-1 uses random_device)\n"
              << "      --temp VALUE         Sampling temperature (default 0.8)\n"
              << "      --top-p VALUE        Top-p nucleus threshold (default 0.95)\n"
              << "      --top-k N            Top-k limit (default 40, 0 = unlimited)\n"
              << "      --repeat-penalty R   Repetition penalty (default 1.1)\n"
              << "      --repeat-last-n N    Window for repetition penalty (default 64)\n"
              << "  -h, --help              Show this help message\n";
}

bool parse_int(const char * arg, const char * value, int32_t & out) {
    if (!value) {
        std::cerr << "Missing value for " << arg << "\n";
        return false;
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0') {
        std::cerr << "Invalid integer for " << arg << ": " << value << "\n";
        return false;
    }
    out = static_cast<int32_t>(parsed);
    return true;
}

bool parse_uint(const char * arg, const char * value, uint32_t & out) {
    if (!value) {
        std::cerr << "Missing value for " << arg << "\n";
        return false;
    }
    char * end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (!end || *end != '\0') {
        std::cerr << "Invalid unsigned integer for " << arg << ": " << value << "\n";
        return false;
    }
    out = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_float(const char * arg, const char * value, float & out) {
    if (!value) {
        std::cerr << "Missing value for " << arg << "\n";
        return false;
    }
    char * end = nullptr;
    float parsed = std::strtof(value, &end);
    if (!end || *end != '\0') {
        std::cerr << "Invalid floating-point value for " << arg << ": " << value << "\n";
        return false;
    }
    out = parsed;
    return true;
}

ggml_numa_strategy parse_numa_mode(const std::string & value) {
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower == "none" || lower == "disabled") {
        return GGML_NUMA_STRATEGY_DISABLED;
    }
    if (lower == "distribute") {
        return GGML_NUMA_STRATEGY_DISTRIBUTE;
    }
    if (lower == "isolate") {
        return GGML_NUMA_STRATEGY_ISOLATE;
    }

    throw std::runtime_error("Unknown NUMA mode: " + value);
}

bool parse_arguments(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](int index) -> const char * {
            if (index >= argc) {
                return nullptr;
            }
            return argv[index];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "-m" || arg == "--model") {
            const char * value = require_value(++i);
            if (!value) {
                std::cerr << "Missing path for --model\n";
                return false;
            }
            opts.model_path = value;
        } else if (arg == "-p" || arg == "--prompt") {
            const char * value = require_value(++i);
            if (!value) {
                std::cerr << "Missing text for --prompt\n";
                return false;
            }
            opts.prompt = value;
        } else if (arg == "-n" || arg == "--n-predict") {
            const char * value = require_value(++i);
            if (!parse_int(arg.c_str(), value, opts.n_predict)) {
                return false;
            }
        } else if (arg == "-t" || arg == "--threads") {
            const char * value = require_value(++i);
            if (!parse_int(arg.c_str(), value, opts.threads)) {
                return false;
            }
        } else if (arg == "-tb" || arg == "--threads-batch") {
            const char * value = require_value(++i);
            if (!parse_int(arg.c_str(), value, opts.threads_batch)) {
                return false;
            }
        } else if (arg == "--ctx-size") {
            const char * value = require_value(++i);
            if (!parse_uint(arg.c_str(), value, opts.context_size)) {
                return false;
            }
        } else if (arg == "--ubatch-size") {
            const char * value = require_value(++i);
            if (!parse_uint(arg.c_str(), value, opts.ubatch_size)) {
                return false;
            }
            opts.batch_size = opts.ubatch_size;
        } else if (arg == "--mlock") {
            opts.use_mlock = true;
        } else if (arg == "--numa") {
            const char * value = require_value(++i);
            if (!value) {
                std::cerr << "Missing value for --numa\n";
                return false;
            }
            try {
                opts.numa = parse_numa_mode(value);
            } catch (const std::exception & ex) {
                std::cerr << ex.what() << "\n";
                return false;
            }
        } else if (arg == "--keep") {
            const char * value = require_value(++i);
            if (!value) {
                std::cerr << "Missing value for --keep\n";
                return false;
            }
            // compatibility option; this minimal CLI always discards the prompt
        } else if (arg == "--seed") {
            const char * value = require_value(++i);
            if (!parse_int(arg.c_str(), value, opts.seed)) {
                return false;
            }
        } else if (arg == "--temp") {
            const char * value = require_value(++i);
            if (!parse_float(arg.c_str(), value, opts.temperature)) {
                return false;
            }
        } else if (arg == "--top-p") {
            const char * value = require_value(++i);
            if (!parse_float(arg.c_str(), value, opts.top_p)) {
                return false;
            }
        } else if (arg == "--top-k") {
            const char * value = require_value(++i);
            if (!parse_int(arg.c_str(), value, opts.top_k)) {
                return false;
            }
        } else if (arg == "--repeat-penalty") {
            const char * value = require_value(++i);
            if (!parse_float(arg.c_str(), value, opts.repeat_penalty)) {
                return false;
            }
        } else if (arg == "--repeat-last-n") {
            const char * value = require_value(++i);
            if (!parse_int(arg.c_str(), value, opts.repeat_last_n)) {
                return false;
            }
        } else {
            std::cerr << "Unrecognized argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    if (opts.model_path.empty()) {
        std::cerr << "A model path must be provided with --model\n";
        print_usage(argv[0]);
        return false;
    }

    if (opts.n_predict == 0) {
        opts.n_predict = 1;
    }

    if (opts.batch_size == 0) {
        opts.batch_size = 1;
    }
    if (opts.ubatch_size == 0) {
        opts.ubatch_size = 1;
    }

    if (opts.temperature < 0.0f) {
        std::cerr << "Temperature cannot be negative\n";
        return false;
    }
    if (opts.top_p <= 0.0f) {
        opts.top_p = 1.0f;
    }
    if (opts.repeat_penalty < 1.0f) {
        opts.repeat_penalty = 1.0f;
    }
    if (opts.repeat_last_n < 0) {
        opts.repeat_last_n = 0;
    }

    return true;
}

std::vector<gptoss_token> tokenize_prompt(const gptoss_vocab * vocab, const std::string & text) {
    if (text.empty()) {
        return { gptoss_vocab_bos(vocab) };
    }

    std::vector<gptoss_token> tokens(text.size() + 8);
    int32_t n_tokens = gptoss_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        /*add_special=*/true,
        /*parse_special=*/true);

    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization result exceeds int32_t limit");
    }

    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = gptoss_tokenize(
            vocab,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            /*add_special=*/true,
            /*parse_special=*/true);
    }

    if (n_tokens < 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }

    tokens.resize(static_cast<size_t>(n_tokens));
    if (tokens.empty()) {
        tokens.push_back(gptoss_vocab_bos(vocab));
    }

    return tokens;
}

std::string token_to_string(const gptoss_vocab * vocab, gptoss_token token) {
    std::vector<char> buffer(64);
    int32_t written = gptoss_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, /*special=*/false);
    if (written < 0) {
        buffer.resize(static_cast<size_t>(-written));
        written = gptoss_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, /*special=*/false);
    }
    if (written < 0) {
        return {};
    }
    return std::string(buffer.data(), buffer.data() + written);
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_arguments(argc, argv, opts)) {
        return 1;
    }

    gptoss_backend_init();
    gptoss_numa_init(opts.numa);

    gptoss_model_params model_params = gptoss_model_default_params();
    model_params.use_mlock = opts.use_mlock;

    gptoss_model * model = gptoss_model_load_from_file(opts.model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model: " << opts.model_path << "\n";
        gptoss_backend_free();
        return 1;
    }

    gptoss_context_params ctx_params = gptoss_context_default_params();
    if (opts.context_size > 0) {
        ctx_params.n_ctx = opts.context_size;
    }
    if (opts.threads > 0) {
        ctx_params.n_threads = opts.threads;
    }
    if (opts.threads_batch > 0) {
        ctx_params.n_threads_batch = opts.threads_batch;
    }
    ctx_params.n_batch  = std::max<uint32_t>(1, opts.batch_size);
    ctx_params.n_ubatch = std::max<uint32_t>(1, opts.ubatch_size);
    ctx_params.n_seq_max = 1;

    gptoss_context * ctx = gptoss_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create inference context\n";
        gptoss_model_free(model);
        gptoss_backend_free();
        return 1;
    }

    const gptoss_vocab * vocab = gptoss_model_get_vocab(model);

    std::vector<gptoss_token> prompt_tokens;
    try {
        prompt_tokens = tokenize_prompt(vocab, opts.prompt);
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << "\n";
        gptoss_free(ctx);
        gptoss_model_free(model);
        gptoss_backend_free();
        return 1;
    }

    const uint32_t batch_tokens = std::max<uint32_t>(1, opts.batch_size);
    size_t processed = 0;
    std::vector<gptoss_token> recent_tokens = prompt_tokens;
    while (processed < prompt_tokens.size()) {
        const uint32_t n_eval = std::min<uint32_t>(batch_tokens, static_cast<uint32_t>(prompt_tokens.size() - processed));
        gptoss_batch batch = gptoss_batch_get_one(prompt_tokens.data() + processed, static_cast<int32_t>(n_eval));
        const int32_t rc = gptoss_decode(ctx, batch);
        if (rc != 0) {
            std::cerr << "gptoss_decode failed during prompt evaluation (code " << rc << ")\n";
            gptoss_free(ctx);
            gptoss_model_free(model);
            gptoss_backend_free();
            return 1;
        }
        processed += n_eval;
    }

    const gptoss_token eos_token = gptoss_vocab_eos(vocab);
    const int32_t vocab_size = gptoss_vocab_n_tokens(vocab);

    int32_t remaining = opts.n_predict < 0 ? std::numeric_limits<int32_t>::max() : opts.n_predict;

    std::mt19937 rng;
    if (opts.seed < 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(static_cast<uint32_t>(opts.seed));
    }

    std::vector<float> logits_buffer(static_cast<size_t>(vocab_size));
    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(vocab_size));

    while (remaining-- > 0) {
        float * logits_raw = gptoss_get_logits(ctx);
        if (!logits_raw) {
            std::cerr << "Failed to obtain logits from context\n";
            break;
        }

        std::memcpy(logits_buffer.data(), logits_raw, static_cast<size_t>(vocab_size) * sizeof(float));

        if (opts.repeat_last_n > 0 && opts.repeat_penalty > 1.0f && !recent_tokens.empty()) {
            const size_t limit = static_cast<size_t>(opts.repeat_last_n);
            const size_t start = recent_tokens.size() > limit ? recent_tokens.size() - limit : 0;
            for (size_t i = start; i < recent_tokens.size(); ++i) {
                const gptoss_token token = recent_tokens[i];
                if (token < 0 || token >= vocab_size) {
                    continue;
                }
                float & logit = logits_buffer[static_cast<size_t>(token)];
                if (logit > 0.0f) {
                    logit /= opts.repeat_penalty;
                } else {
                    logit *= opts.repeat_penalty;
                }
            }
        }

        gptoss_token next = eos_token;

        if (opts.temperature <= 0.0f) {
            float best_value = logits_buffer[0];
            int best_index = 0;
            for (int i = 1; i < vocab_size; ++i) {
                const float value = logits_buffer[static_cast<size_t>(i)];
                if (value > best_value) {
                    best_value = value;
                    best_index = i;
                }
            }
            next = static_cast<gptoss_token>(best_index);
        } else {
            candidates.clear();
            const int effective_top_k = (opts.top_k > 0 && opts.top_k < vocab_size) ? opts.top_k : vocab_size;
            candidates.reserve(static_cast<size_t>(effective_top_k));

            if (effective_top_k < vocab_size) {
                std::vector<int> indices(static_cast<size_t>(vocab_size));
                std::iota(indices.begin(), indices.end(), 0);
                std::nth_element(
                    indices.begin(),
                    indices.begin() + effective_top_k,
                    indices.end(),
                    [&](int a, int b) {
                        return logits_buffer[static_cast<size_t>(a)] > logits_buffer[static_cast<size_t>(b)];
                    });
                indices.resize(static_cast<size_t>(effective_top_k));
                candidates = std::move(indices);
            } else {
                candidates.resize(static_cast<size_t>(vocab_size));
                std::iota(candidates.begin(), candidates.end(), 0);
            }

            std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                return logits_buffer[static_cast<size_t>(a)] > logits_buffer[static_cast<size_t>(b)];
            });

            const float inv_temp = 1.0f / opts.temperature;
            std::vector<float> probabilities;
            probabilities.reserve(candidates.size());

            float max_logit = logits_buffer[static_cast<size_t>(candidates.front())];
            float normalizer = 0.0f;
            for (int idx : candidates) {
                float prob = std::exp((logits_buffer[static_cast<size_t>(idx)] - max_logit) * inv_temp);
                probabilities.push_back(prob);
                normalizer += prob;
            }

            if (normalizer <= 0.0f) {
                next = static_cast<gptoss_token>(candidates.front());
            } else {
                float cutoff = opts.top_p >= 1.0f ? std::numeric_limits<float>::infinity() : opts.top_p;
                float cumulative = 0.0f;
                std::vector<std::pair<int, float>> filtered;
                filtered.reserve(probabilities.size());

                for (size_t i = 0; i < probabilities.size(); ++i) {
                    float prob = probabilities[i] / normalizer;
                    cumulative += prob;
                    filtered.emplace_back(candidates[i], prob);
                    if (cumulative >= cutoff) {
                        break;
                    }
                }

                float draw = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
                float running = 0.0f;
                for (const auto & entry : filtered) {
                    running += entry.second;
                    if (draw <= running || &entry == &filtered.back()) {
                        next = static_cast<gptoss_token>(entry.first);
                        break;
                    }
                }
            }
        }

        if (next == eos_token) {
            break;
        }

        std::cout << token_to_string(vocab, next);
        std::cout.flush();

        recent_tokens.push_back(next);

        gptoss_batch batch = gptoss_batch_get_one(&next, 1);
        const int32_t rc = gptoss_decode(ctx, batch);
        if (rc != 0) {
            std::cerr << "\nGeneration failed with code " << rc << "\n";
            break;
        }
    }

    std::cout << std::endl;

    gptoss_free(ctx);
    gptoss_model_free(model);
    gptoss_backend_free();
    return 0;
}
