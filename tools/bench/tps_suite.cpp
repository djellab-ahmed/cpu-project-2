// tools/bench/tps_suite.cpp
// Drives a fixed set of 10 prompts end-to-end and aggregates "True TPS":
//   True TPS = (sum generated_tokens) / (sum elapsed_seconds)
// Uses the repo's CLI binary (gptoss-cli) so we don't couple to internal headers.
// It also merges per-token JSON traces that gptoss-cli emits via --bench-json.
//
// Build target added below in CMake. Usage:
//   ./build/bin/tps-suite \
//      --cli ./build/bin/gptoss-cli \
//      --model models/gpt-oss-20b-Q4_K_M.gguf \
//      --threads 16 --n-predict 256 \
//      --prompts tools/bench/prompts10.txt \
//      --out tools/bench/logs/suite.json
//
// Notes:
// - If you prefer to link internal APIs, replace run_one() to call directly into your
//   context instead of shelling to gptoss-cli.

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct RunCfg {
    std::string cli;
    std::string model;
    std::string prompts_file;
    std::string out_json;
    int threads = 16;
    int n_predict = 256;
    std::string extra;
};

static std::vector<std::string> read_prompts(const std::string & path) {
    std::ifstream f(path);
    std::vector<std::string> prompts;
    std::string line;
    std::ostringstream cur;
    while (std::getline(f, line)) {
        if (!line.empty() && line.rfind("###", 0) == 0) {
            if (cur.tellp() > 0) {
                prompts.emplace_back(cur.str());
                cur.str("");
                cur.clear();
            }
            continue;
        }
        cur << line << '\n';
    }
    if (cur.tellp() > 0) {
        prompts.emplace_back(cur.str());
    }
    return prompts;
}

static int run_one(const RunCfg & cfg,
                   const std::string & prompt,
                   int index,
                   double & elapsed_s,
                   int & tokens_gen,
                   std::string & trace_path) {
    char tmp_name[] = "/tmp/promptXXXXXX.txt";
    int fd = mkstemps(tmp_name, 4);
    if (fd < 0) {
        perror("mkstemps");
        return -1;
    }
    FILE * pf = fdopen(fd, "w");
    if (!pf) {
        ::close(fd);
        perror("fdopen");
        return -1;
    }
    fwrite(prompt.data(), 1, prompt.size(), pf);
    fclose(pf);

    std::ostringstream tp;
    tp << "/tmp/bench_trace_" << index << ".json";
    trace_path = tp.str();

    std::vector<std::string> args;
    args.push_back(cfg.cli);
    args.push_back("-m");
    args.push_back(cfg.model);
    args.push_back("-t");
    args.push_back(std::to_string(cfg.threads));
    args.push_back("-n");
    args.push_back(std::to_string(cfg.n_predict));
    args.push_back("--prompt-file");
    args.push_back(tmp_name);
    args.push_back("--bench-json");
    args.push_back(trace_path);
    if (!cfg.extra.empty()) {
        args.push_back(cfg.extra);
    }
    args.push_back("--");

    std::vector<char *> argv;
    for (auto & s : args) {
        argv.push_back(const_cast<char *>(s.c_str()));
    }
    argv.push_back(nullptr);

    auto t0 = std::chrono::steady_clock::now();
    pid_t pid = fork();
    if (pid == 0) {
        execv(argv[0], argv.data());
        perror("execv");
        _exit(127);
    } else if (pid < 0) {
        perror("fork");
        unlink(tmp_name);
        return -1;
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        unlink(tmp_name);
        return -1;
    }
    auto t1 = std::chrono::steady_clock::now();
    elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    std::ifstream jf(trace_path);
    tokens_gen = 0;
    if (jf) {
        std::string json((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
        auto pos = json.find("\"tokens_generated\":");
        if (pos != std::string::npos) {
            tokens_gen = std::atoi(json.c_str() + pos + 20);
        }
    }
    unlink(tmp_name);

    if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
        return -1;
    }
    return 0;
}

static void write_suite_json(const std::string & path,
                             const std::vector<double> & elapsed,
                             const std::vector<int> & tokens,
                             const std::vector<std::string> & traces) {
    double sum_elapsed = 0.0;
    long sum_tokens = 0;
    for (size_t i = 0; i < elapsed.size(); ++i) {
        sum_elapsed += elapsed[i];
        sum_tokens += tokens[i];
    }
    double tps = sum_elapsed > 0.0 ? static_cast<double>(sum_tokens) / sum_elapsed : 0.0;

    std::ofstream f(path);
    f << "{\n";
    f << "  \"suite\": {\n";
    f << "    \"prompts\": " << elapsed.size() << ",\n";
    f << "    \"sum_elapsed_s\": " << sum_elapsed << ",\n";
    f << "    \"sum_generated_tokens\": " << sum_tokens << ",\n";
    f << "    \"true_tps\": " << tps << "\n";
    f << "  },\n";
    f << "  \"runs\": [\n";
    for (size_t i = 0; i < elapsed.size(); ++i) {
        f << "    {\"index\": " << i
          << ", \"elapsed_s\": " << elapsed[i]
          << ", \"generated_tokens\": " << tokens[i]
          << ", \"trace\": \"" << traces[i] << "\"}";
        if (i + 1 != elapsed.size()) {
            f << ",";
        }
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

int main(int argc, char ** argv) {
    RunCfg cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--cli") {
            cfg.cli = require_value(arg.c_str());
        } else if (arg == "--model") {
            cfg.model = require_value(arg.c_str());
        } else if (arg == "--threads") {
            cfg.threads = std::atoi(require_value(arg.c_str()));
        } else if (arg == "--n-predict") {
            cfg.n_predict = std::atoi(require_value(arg.c_str()));
        } else if (arg == "--prompts") {
            cfg.prompts_file = require_value(arg.c_str());
        } else if (arg == "--out") {
            cfg.out_json = require_value(arg.c_str());
        } else if (arg == "--extra") {
            cfg.extra = require_value(arg.c_str());
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            return 2;
        }
    }

    if (cfg.cli.empty() || cfg.model.empty() || cfg.prompts_file.empty() || cfg.out_json.empty()) {
        std::cerr << "usage: " << argv[0]
                  << " --cli ./build/bin/gptoss-cli"
                  << " --model models/gpt-oss-20b-Q4_K_M.gguf"
                  << " --threads 16 --n-predict 256"
                  << " --prompts tools/bench/prompts10.txt"
                  << " --out tools/bench/logs/suite.json\n";
        return 2;
    }

    auto prompts = read_prompts(cfg.prompts_file);
    if (prompts.size() != 10) {
        std::cerr << "expected exactly 10 prompts in " << cfg.prompts_file
                  << " separated by lines starting with ###\n";
    }

    std::vector<double> elapsed(prompts.size(), 0.0);
    std::vector<int> tokens(prompts.size(), 0);
    std::vector<std::string> traces(prompts.size());

    for (size_t i = 0; i < prompts.size(); ++i) {
        if (run_one(cfg, prompts[i], static_cast<int>(i), elapsed[i], tokens[i], traces[i]) != 0) {
            std::cerr << "run " << i << " failed\n";
        }
    }

    write_suite_json(cfg.out_json, elapsed, tokens, traces);
    std::cout << "Suite true TPS report written to: " << cfg.out_json << "\n";
    return 0;
}
