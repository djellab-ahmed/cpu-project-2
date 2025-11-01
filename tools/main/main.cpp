#include "gptoss.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
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
    int32_t     n_predict      = -1;
    bool        use_mlock      = false;
    ggml_numa_strategy numa    = GGML_NUMA_STRATEGY_DISABLED;
    int32_t     seed           = -1;
    float       temperature    = 0.8f;
    float       top_p          = 0.95f;
    int32_t     top_k          = 40;
    float       repeat_penalty = 1.1f;
    int32_t     repeat_last_n  = 64;
    bool        measure_tps    = false;
};

void print_usage(const char * program) {
    std::cerr << "Usage: " << program << " [options]\n"
              << "\nRequired:\n"
              << "  -m, --model PATH         Path to the GGUF model file\n"
              << "\nOptional:\n"
              << "  -p, --prompt TEXT        Text prompt to evaluate\n"
              << "  -n, --n-predict N        Number of tokens to generate (default unlimited; set -1 for EOS-driven)\n"
              << "      --measure-tps        Run the 10 standard prompts and report True TPS statistics\n"
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
        } else if (arg == "--measure-tps") {
            opts.measure_tps = true;
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

struct generation_metrics {
    size_t generated_tokens = 0;
    double elapsed_seconds  = 0.0;
};

bool run_generation(const options & opts, gptoss_model * model, const std::string & prompt, bool stream_tokens, generation_metrics & metrics) {
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
        return false;
    }

    const gptoss_vocab * vocab = gptoss_model_get_vocab(model);

    std::vector<gptoss_token> prompt_tokens;
    try {
        prompt_tokens = tokenize_prompt(vocab, prompt);
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << "\n";
        gptoss_free(ctx);
        return false;
    }

    const uint32_t batch_tokens = std::max<uint32_t>(1, opts.batch_size);
    size_t processed = 0;
    const size_t repeat_window = opts.repeat_last_n > 0 ? static_cast<size_t>(opts.repeat_last_n) : 0;
    std::deque<gptoss_token> recent_tokens;
    if (repeat_window > 0 && !prompt_tokens.empty()) {
        const size_t take = std::min(repeat_window, prompt_tokens.size());
        recent_tokens.insert(
            recent_tokens.end(),
            prompt_tokens.end() - static_cast<std::ptrdiff_t>(take),
            prompt_tokens.end());
    }

    auto start_time = std::chrono::steady_clock::now();

    while (processed < prompt_tokens.size()) {
        const uint32_t n_eval = std::min<uint32_t>(batch_tokens, static_cast<uint32_t>(prompt_tokens.size() - processed));
        gptoss_batch batch = gptoss_batch_get_one(prompt_tokens.data() + processed, static_cast<int32_t>(n_eval));
        const int32_t rc = gptoss_decode(ctx, batch);
        if (rc != 0) {
            std::cerr << "gptoss_decode failed during prompt evaluation (code " << rc << ")\n";
            gptoss_free(ctx);
            return false;
        }
        processed += n_eval;
    }

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

    size_t generated_tokens = 0;

    while (remaining-- > 0) {
        float * logits_raw = gptoss_get_logits(ctx);
        if (!logits_raw) {
            std::cerr << "Failed to obtain logits from context\n";
            break;
        }

        std::memcpy(logits_buffer.data(), logits_raw, static_cast<size_t>(vocab_size) * sizeof(float));

        if (repeat_window > 0 && opts.repeat_penalty > 1.0f && !recent_tokens.empty()) {
            for (const gptoss_token token : recent_tokens) {
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

        gptoss_token next = gptoss_vocab_eos(vocab);

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

        if (gptoss_vocab_is_eog(vocab, next)) {
            break;
        }

        if (stream_tokens) {
            std::cout << token_to_string(vocab, next);
            std::cout.flush();
        }

        ++generated_tokens;

        if (repeat_window > 0) {
            if (recent_tokens.size() == repeat_window) {
                recent_tokens.pop_front();
            }
            recent_tokens.push_back(next);
        }

        gptoss_batch batch = gptoss_batch_get_one(&next, 1);
        const int32_t rc = gptoss_decode(ctx, batch);
        if (rc != 0) {
            std::cerr << "\nGeneration failed with code " << rc << "\n";
            break;
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    if (stream_tokens) {
        std::cout << std::endl;
    }

    metrics.generated_tokens = generated_tokens;
    metrics.elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    gptoss_free(ctx);
    return true;
}

bool run_measurement(const options & opts, gptoss_model * model) {
    static const char * prompts[10] = {
        "What is AI? Explain artificial intelligence, machine learning, and deep learning. Describe their differences and applications in detail.",
        "Write a comprehensive essay about the history of computing, covering major milestones from the abacus to modern quantum computers. Include key inventors, breakthrough technologies, and the impact on society.",
        "Discuss the evolution of programming languages, operating systems, and computer architectures.",
        "Summarize the differences between inference, training, and fine-tuning for large language models in exactly 6 bullet points. Each bullet ≤25 words.",
        "Write a robust Python script that reads NDJSON from stdin, filters records where rule_level ≥10 or mitreid is not null, outputs NDJSON, and includes argparse, logging, and unit tests.",
        "Produce a 900–1,200 word mini-report: “History of Transformer Models (2017–2025)”. Use section headers, numbered references, and a concluding “Key Takeaways” list of 10 items.",
        "Translate this to English and explain it for a non-technical audience in 5 bullets: “Les compteurs intelligents doivent conserver au moins 13 mois de profils de charge et journaliser les événements de fraude.”",
        "Return ONLY valid JSON summarizing pros/cons of CPU-only vs GPU inference for a 20B parameter model. Fields: {\"hardware\": [...], \"throughput_tpm\": \"string\", \"latency_ms\": \"string\", \"energy_tradeoffs\": [...], \"when_to_choose\": {\"cpu\": [...], \"gpu\": [...]}}.",
        "Write a detailed design doc (~800 words) for a PostgreSQL + TimescaleDB pipeline that ingests 4B log rows/day, with partitioning, continuous aggregates, and a watermarked incremental job. Include sample DDL and 3 optimized queries.",
        "Generate a step-by-step tutorial that builds a retrieval-augmented generation (RAG) prototype with Ollama + FAISS. Include: environment setup, data ingestion, embedding choice rationale, retrieval API, evaluation checklist, and a final “Gotchas” section."
    };

    std::vector<generation_metrics> results;
    results.resize(10);

    size_t total_tokens = 0;
    double total_seconds = 0.0;

    for (size_t i = 0; i < results.size(); ++i) {
        generation_metrics metrics;
        if (!run_generation(opts, model, prompts[i], /*stream_tokens=*/false, metrics)) {
            return false;
        }
        results[i] = metrics;
        total_tokens += metrics.generated_tokens;
        total_seconds += metrics.elapsed_seconds;
    }

    std::cout << "-----------------------------------------------\n";
    std::cout << "PROMPT  | TOKENS | TIME (s) | TPS\n";
    std::cout << "-----------------------------------------------\n";

    std::cout << std::fixed << std::setprecision(1);

    for (size_t i = 0; i < results.size(); ++i) {
        const double time_s = results[i].elapsed_seconds;
        const double tps = time_s > 0.0 ? static_cast<double>(results[i].generated_tokens) / time_s : 0.0;
        std::cout << std::setw(7) << std::left << (i + 1)
                  << " | " << std::setw(6) << std::right << results[i].generated_tokens << " | "
                  << std::setw(8) << std::right << time_s << " | "
                  << std::setw(5) << std::right << tps << "\n";
    }

    std::cout << "-----------------------------------------------\n";
    const double true_tps = total_seconds > 0.0 ? static_cast<double>(total_tokens) / total_seconds : 0.0;
    std::cout << std::setw(7) << std::left << "TOTAL"
              << " | " << std::setw(6) << std::right << total_tokens << " | "
              << std::setw(8) << std::right << total_seconds << " | "
              << std::setw(5) << std::right << true_tps << "  (TRUE TPS)\n";
    std::cout << "-----------------------------------------------" << std::endl;

    std::cout.unsetf(std::ios::floatfield);

    return true;
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
    int exit_code = 0;

    if (opts.measure_tps) {
        if (!run_measurement(opts, model)) {
            exit_code = 1;
        }
    } else {
        generation_metrics metrics;
        if (!run_generation(opts, model, opts.prompt, /*stream_tokens=*/true, metrics)) {
            exit_code = 1;
        }
    }

    gptoss_model_free(model);
    gptoss_backend_free();
    return exit_code;
}
