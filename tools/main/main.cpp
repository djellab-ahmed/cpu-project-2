#include "gptoss.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdio>
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
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
#if defined(GGML_BLAS_USE_MKL)
    void mkl_set_num_threads_local(int);
#endif
#if defined(OPENBLAS_VERSION)
    void openblas_set_num_threads(int);
#endif
#if defined(GGML_BLAS_USE_BLIS)
    void bli_thread_set_num_threads(int);
#endif
}

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
    bool        quiet_mode     = false;
    bool        bench_mode     = false;
    bool        kv_q8          = false;
    std::string kv_q8_scheme   = "row,row";
};

void print_usage(const char * program) {
    std::cerr << "Usage: " << program << " [options]\n"
              << "\nRequired:\n"
              << "  -m, --model PATH         Path to the GGUF model file\n"
              << "\nOptional:\n"
              << "  -p, --prompt TEXT        Text prompt to evaluate\n"
              << "  -n, --n-predict N        Number of tokens to generate (default unlimited; set -1 for EOS-driven)\n"
              << "      --measure-tps        Run the 10 standard prompts and report True TPS statistics\n"
              << "      --bench             Emit BENCH timing CSV for the provided prompt\n"
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
              << "      --kv-q8              Enable experimental INT8 KV cache\n"
              << "      --kv-q8-scheme NAME  INT8 KV cache scheme (default row,row)\n"
              << "      --quiet              Suppress streamed token output\n"
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
        } else if (arg == "--kv-q8") {
            opts.kv_q8 = true;
        } else if (arg == "--kv-q8-scheme") {
            const char * value = require_value(++i);
            if (!value) {
                std::cerr << "Missing value for --kv-q8-scheme\n";
                return false;
            }
            opts.kv_q8_scheme = value;
            if (opts.kv_q8_scheme != "row,row") {
                std::cerr << "Unsupported --kv-q8-scheme: " << opts.kv_q8_scheme << "\n";
                return false;
            }
        } else if (arg == "--measure-tps") {
            opts.measure_tps = true;
        } else if (arg == "--quiet") {
            opts.quiet_mode = true;
        } else if (arg == "--bench") {
            opts.bench_mode = true;
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

void filtered_log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
        case GGML_LOG_LEVEL_WARN:
            std::fwrite(text, 1, std::strlen(text), stderr);
            std::fflush(stderr);
            break;
        default:
            break;
    }
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
    double prefill_ms       = 0.0;
    double decode_ms        = 0.0;
};

bool run_generation(const options & opts, gptoss_model * model, const std::string & prompt, bool stream_tokens, generation_metrics & metrics) {
    const gptoss_vocab * vocab = gptoss_model_get_vocab(model);

    std::vector<gptoss_token> prompt_tokens;
    try {
        prompt_tokens = tokenize_prompt(vocab, prompt);
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << "\n";
        return false;
    }

    gptoss_context_params ctx_params = gptoss_context_default_params();
    const uint32_t prompt_len = static_cast<uint32_t>(prompt_tokens.size());
    uint32_t desired_ctx = opts.context_size;

    if (desired_ctx == 0) {
        const uint32_t predictive_headroom = opts.n_predict > 0 ? static_cast<uint32_t>(opts.n_predict) : 2048u;
        desired_ctx = prompt_len + predictive_headroom;
        desired_ctx = std::max<uint32_t>(desired_ctx, 8192u);
        const int32_t model_ctx = gptoss_model_n_ctx_train(model);
        if (model_ctx > 0) {
            desired_ctx = std::min<uint32_t>(desired_ctx, static_cast<uint32_t>(model_ctx));
        }
    }

    ctx_params.n_ctx = desired_ctx;

    const uint32_t hw_threads = std::max(1u, std::thread::hardware_concurrency());
    ctx_params.n_threads = opts.threads > 0 ? opts.threads : static_cast<int32_t>(hw_threads);
    ctx_params.n_threads_batch = opts.threads_batch > 0 ? opts.threads_batch : static_cast<int32_t>(hw_threads);

    const uint32_t batch_limit = std::max<uint32_t>(1, opts.batch_size);
    const uint32_t ubatch_limit = std::max<uint32_t>(1, opts.ubatch_size);
    ctx_params.n_batch  = std::min<uint32_t>(ctx_params.n_ctx, batch_limit);
    ctx_params.n_ubatch = std::min<uint32_t>(ctx_params.n_ctx, ubatch_limit);
    ctx_params.n_seq_max = 1;
    ctx_params.kv_cache_q8 = opts.kv_q8;
    std::snprintf(ctx_params.kv_cache_q8_scheme, sizeof(ctx_params.kv_cache_q8_scheme), "%s", opts.kv_q8_scheme.c_str());

    gptoss_context * ctx = gptoss_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create inference context\n";
        return false;
    }

    const uint32_t batch_tokens = ctx_params.n_batch;
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
    auto prefill_start = start_time;

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

    auto prefill_end = std::chrono::steady_clock::now();
    metrics.prefill_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(prefill_end - prefill_start).count();

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

    auto decode_start = prefill_end;

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
    metrics.decode_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - decode_start).count();

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
        std::string preview = prompts[i];
        for (char & ch : preview) {
            if (ch == '\n' || ch == '\r' || ch == '\t') {
                ch = ' ';
            }
        }
        const size_t max_preview = 72;
        if (preview.size() > max_preview) {
            preview = preview.substr(0, max_preview - 3) + "...";
        }

        std::cout << "\n[Benchmark] (" << (i + 1) << "/" << results.size() << ") "
                  << preview << std::endl;

        generation_metrics metrics;
        if (!run_generation(opts, model, prompts[i], /*stream_tokens=*/false, metrics)) {
            return false;
        }
        results[i] = metrics;
        total_tokens += metrics.generated_tokens;
        total_seconds += metrics.elapsed_seconds;

        std::ostringstream summary;
        summary.setf(std::ios::fixed, std::ios::floatfield);
        summary << std::setprecision(2);
        const double time_s = metrics.elapsed_seconds;
        const double tps = time_s > 0.0 ? static_cast<double>(metrics.generated_tokens) / time_s : 0.0;
        summary << "    -> tokens: " << metrics.generated_tokens
                << ", time: " << time_s << " s, TPS: " << tps;
        std::cout << summary.str() << std::endl;
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

#ifdef _OPENMP
    if (opts.threads > 0) {
        omp_set_num_threads(opts.threads);
    }
#endif
#if defined(GGML_BLAS_USE_MKL)
    mkl_set_num_threads_local(1);
#endif
#if defined(OPENBLAS_VERSION)
    openblas_set_num_threads(1);
#endif
#if defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_num_threads(1);
#endif

    gptoss_log_set(filtered_log_callback, nullptr);
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
        const bool stream_tokens = !opts.quiet_mode && !opts.bench_mode;
        if (!run_generation(opts, model, opts.prompt, stream_tokens, metrics)) {
            exit_code = 1;
        } else if (opts.bench_mode) {
            std::ostringstream bench_line;
            bench_line.setf(std::ios::fixed, std::ios::floatfield);
            bench_line << std::setprecision(3)
                       << "BENCH,P_MS=" << metrics.prefill_ms
                       << ",D_MS=" << metrics.decode_ms
                       << ",TOK=" << metrics.generated_tokens;
            std::cout << bench_line.str() << std::endl;
        }
    }

    gptoss_model_free(model);
    gptoss_backend_free();
    return exit_code;
}
