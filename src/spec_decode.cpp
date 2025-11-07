#include "spec_decode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

namespace {

constexpr float kEpsilon = 1e-6f;

bool snapshot_sequence(gptoss_context * ctx, std::vector<uint8_t> & buffer) {
    if (!ctx) {
        return false;
    }
    const gptoss_seq_id seq_id = 0;
    size_t size = gptoss_state_seq_get_size(ctx, seq_id);
    if (size == 0) {
        buffer.clear();
        return true;
    }
    if (buffer.size() < size) {
        buffer.resize(size);
    }
    size_t written = gptoss_state_seq_get_data(ctx, buffer.data(), buffer.size(), seq_id);
    return written == size;
}

bool restore_sequence(gptoss_context * ctx, const std::vector<uint8_t> & buffer) {
    if (!ctx) {
        return false;
    }
    if (buffer.empty()) {
        return true;
    }
    const gptoss_seq_id seq_id = 0;
    size_t read = gptoss_state_seq_set_data(ctx, buffer.data(), buffer.size(), seq_id);
    return read != 0;
}

bool decode_single(gptoss_context * ctx, gptoss_token token) {
    if (!ctx) {
        return false;
    }
    gptoss_batch batch = gptoss_batch_get_one(&token, 1);
    const int32_t rc = gptoss_decode(ctx, batch);
    return rc == 0;
}

void apply_repetition_penalty(
        std::vector<float> & logits,
        const std::deque<gptoss_token> & window,
        float repeat_penalty) {
    if (repeat_penalty <= 1.0f || window.empty()) {
        return;
    }
    const int vocab_size = static_cast<int>(logits.size());
    for (gptoss_token token : window) {
        if (token < 0 || token >= vocab_size) {
            continue;
        }
        float & value = logits[static_cast<size_t>(token)];
        if (value > 0.0f) {
            value /= repeat_penalty;
        } else {
            value *= repeat_penalty;
        }
    }
}

void push_repetition(std::deque<gptoss_token> & window, gptoss_token token, size_t limit) {
    if (limit == 0) {
        return;
    }
    if (window.size() == limit) {
        window.pop_front();
    }
    window.push_back(token);
}

void build_adjusted_probs(
        const float * logits,
        int vocab_size,
        float temperature,
        int top_k,
        float top_p,
        const std::deque<gptoss_token> & repeat_window,
        float repeat_penalty,
        std::vector<float> & probs,
        std::vector<int> & order,
        std::vector<int> & scratch) {
    probs.assign(logits, logits + vocab_size);

    apply_repetition_penalty(probs, repeat_window, repeat_penalty);

    order.clear();
    scratch.resize(static_cast<size_t>(vocab_size));
    std::iota(scratch.begin(), scratch.end(), 0);

    if (temperature <= 0.0f) {
        const auto it = std::max_element(probs.begin(), probs.end());
        int best = static_cast<int>(std::distance(probs.begin(), it));
        std::fill(probs.begin(), probs.end(), 0.0f);
        probs[static_cast<size_t>(best)] = 1.0f;
        order.push_back(best);
        return;
    }

    const int effective_top_k = (top_k > 0 && top_k < vocab_size) ? top_k : vocab_size;
    if (effective_top_k < vocab_size) {
        std::nth_element(
                scratch.begin(),
                scratch.begin() + effective_top_k,
                scratch.end(),
                [&](int a, int b) {
                    return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)];
                });
        scratch.resize(static_cast<size_t>(effective_top_k));
    }

    std::sort(
            scratch.begin(),
            scratch.end(),
            [&](int a, int b) {
                return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)];
            });

    if (scratch.empty()) {
        order.clear();
        probs.assign(static_cast<size_t>(vocab_size), 0.0f);
        return;
    }

    const float inv_temp = 1.0f / temperature;
    const float max_logit = probs[static_cast<size_t>(scratch.front())];

    std::vector<float> normed;
    normed.reserve(scratch.size());
    float normalizer = 0.0f;
    for (int idx : scratch) {
        const float value = std::exp((probs[static_cast<size_t>(idx)] - max_logit) * inv_temp);
        normed.push_back(value);
        normalizer += value;
    }

    std::fill(probs.begin(), probs.end(), 0.0f);

    if (normalizer <= 0.0f) {
        int chosen = scratch.front();
        probs[static_cast<size_t>(chosen)] = 1.0f;
        order.assign(1, chosen);
        return;
    }

    const float cutoff = top_p >= 1.0f ? std::numeric_limits<float>::infinity() : top_p;
    float cumulative = 0.0f;
    order.clear();
    order.reserve(normed.size());

    std::vector<float> filtered_probs;
    filtered_probs.reserve(normed.size());

    for (size_t i = 0; i < normed.size(); ++i) {
        const float normalized = normed[i] / normalizer;
        const int idx = scratch[i];
        order.push_back(idx);
        filtered_probs.push_back(normalized);
        cumulative += normalized;
        if (cumulative >= cutoff) {
            break;
        }
    }

    float leftover = 1.0f;
    for (size_t i = 0; i < order.size(); ++i) {
        float value = filtered_probs[i];
        if (i + 1 == order.size()) {
            value = std::max(0.0f, leftover);
        } else {
            leftover -= value;
        }
        probs[static_cast<size_t>(order[i])] = value;
    }
}

void enforce_greedy_delta(
        std::vector<float> & probs,
        std::vector<int> & order) {
    if (probs.empty()) {
        return;
    }

    int best = -1;
    if (!order.empty()) {
        best = order.front();
    } else {
        auto it = std::max_element(probs.begin(), probs.end());
        if (it != probs.end()) {
            best = static_cast<int>(std::distance(probs.begin(), it));
        }
    }

    if (best < 0 || best >= static_cast<int>(probs.size())) {
        return;
    }

    std::fill(probs.begin(), probs.end(), 0.0f);
    probs[static_cast<size_t>(best)] = 1.0f;
    order.assign(1, best);
}

gptoss_token sample_from_probs(
        const std::vector<float> & probs,
        const std::vector<int> & order,
        std::mt19937 & rng) {
    if (order.empty()) {
        return 0;
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float draw = dist(rng);
    float running = 0.0f;
    for (size_t i = 0; i < order.size(); ++i) {
        const int idx = order[i];
        const float prob = probs[static_cast<size_t>(idx)];
        running += prob;
        if (draw <= running || i + 1 == order.size()) {
            return static_cast<gptoss_token>(idx);
        }
    }
    return static_cast<gptoss_token>(order.back());
}

} // namespace

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
        std::mt19937 & rng) {
    if (!sx.verifier_ctx || !sx.draft_ctx || !sx.verifier_model || !sx.draft_model) {
        return 0;
    }

    const gptoss_vocab * vocab = gptoss_model_get_vocab(sx.verifier_model);
    if (!vocab) {
        return 0;
    }
    const int vocab_size = gptoss_vocab_n_tokens(vocab);
    if (vocab_size <= 0) {
        return 0;
    }

    const size_t repeat_window = repeat_last_n > 0 ? static_cast<size_t>(repeat_last_n) : 0;
    std::deque<gptoss_token> base_window;
    if (repeat_window > 0 && !recent_for_repetition.empty()) {
        const size_t start = recent_for_repetition.size() > repeat_window
                ? recent_for_repetition.size() - repeat_window
                : 0;
        for (size_t i = start; i < recent_for_repetition.size(); ++i) {
            base_window.push_back(recent_for_repetition[i]);
        }
    }

    const int max_propose = std::max(1, cfg.max_propose);
    sx.proposals.clear();
    sx.q_token_prob.clear();
    sx.p_token_prob.clear();
    sx.proposals.reserve(static_cast<size_t>(max_propose));
    sx.q_token_prob.reserve(static_cast<size_t>(max_propose));
    sx.p_token_prob.reserve(static_cast<size_t>(max_propose));

    sx.p_next.resize(static_cast<size_t>(vocab_size));
    sx.q_next.resize(static_cast<size_t>(vocab_size));

    if (!snapshot_sequence(sx.draft_ctx, sx.draft_state)) {
        return 0;
    }

    std::deque<gptoss_token> draft_window = base_window;
    std::mt19937 draft_rng = rng;

    bool encountered_eos = false;

    for (int i = 0; i < max_propose; ++i) {
        const float * logits_raw = gptoss_get_logits(sx.draft_ctx);
        if (!logits_raw) {
            break;
        }

        build_adjusted_probs(
                logits_raw,
                vocab_size,
                temperature,
                top_k,
                top_p,
                draft_window,
                repeat_penalty,
                sx.q_next,
                sx.candidate_order,
                sx.candidate_scratch);

        if (cfg.greedy_draft) {
            enforce_greedy_delta(sx.q_next, sx.candidate_order);
        }

        if (sx.candidate_order.empty()) {
            break;
        }

        gptoss_token proposal = 0;
        float proposal_prob = 1.0f;
        if (cfg.greedy_draft) {
            proposal = static_cast<gptoss_token>(sx.candidate_order.front());
            proposal_prob = 1.0f;
        } else {
            proposal = sample_from_probs(sx.q_next, sx.candidate_order, draft_rng);
            const size_t idx = static_cast<size_t>(proposal);
            if (proposal < 0 || proposal >= vocab_size) {
                proposal_prob = 0.0f;
            } else {
                proposal_prob = std::max(sx.q_next[idx], kEpsilon);
            }
        }

        sx.proposals.push_back(static_cast<int>(proposal));
        sx.q_token_prob.push_back(proposal_prob);

        if (!decode_single(sx.draft_ctx, proposal)) {
            break;
        }

        if (repeat_window > 0) {
            push_repetition(draft_window, proposal, repeat_window);
        }

        if (gptoss_vocab_is_eog(vocab, proposal)) {
            encountered_eos = true;
            break;
        }
    }

    const int proposals_count = static_cast<int>(sx.proposals.size());

    if (!restore_sequence(sx.draft_ctx, sx.draft_state)) {
        return 0;
    }

    if (proposals_count == 0) {
        const float * logits_raw = gptoss_get_logits(sx.verifier_ctx);
        if (!logits_raw) {
            return 0;
        }
        build_adjusted_probs(
                logits_raw,
                vocab_size,
                temperature,
                top_k,
                top_p,
                base_window,
                repeat_penalty,
                sx.p_next,
                sx.candidate_order,
                sx.candidate_scratch);

        const gptoss_token fallback = sample_from_probs(sx.p_next, sx.candidate_order, rng);
        if (!decode_single(sx.verifier_ctx, fallback)) {
            return 0;
        }
        if (!decode_single(sx.draft_ctx, fallback)) {
            return 0;
        }
        out_tokens.push_back(fallback);
        mx.steps++;
        mx.proposed += 1;
        const double ratio = 0.0;
        mx.ewma_accept = mx.steps == 1 ? ratio : 0.95 * mx.ewma_accept + 0.05 * ratio;
        if (cfg.debug) {
            std::cerr << "[spec] fallback step appended=1" << std::endl;
        }
        return 1;
    }

    if (!snapshot_sequence(sx.verifier_ctx, sx.verifier_state)) {
        return 0;
    }

    std::deque<gptoss_token> verifier_window = base_window;
    std::mt19937 acceptance_rng = rng;

    int accepted = 0;
    bool rejected = false;
    gptoss_token eos_token = GPTOSS_TOKEN_NULL;

    for (int i = 0; i < proposals_count; ++i) {
        const float * logits_raw = gptoss_get_logits(sx.verifier_ctx);
        if (!logits_raw) {
            rejected = true;
            break;
        }

        build_adjusted_probs(
                logits_raw,
                vocab_size,
                temperature,
                top_k,
                top_p,
                verifier_window,
                repeat_penalty,
                sx.p_next,
                sx.candidate_order,
                sx.candidate_scratch);

        const gptoss_token proposal = static_cast<gptoss_token>(sx.proposals[static_cast<size_t>(i)]);
        float pi = 0.0f;
        if (proposal >= 0 && proposal < vocab_size) {
            pi = sx.p_next[static_cast<size_t>(proposal)];
        }
        sx.p_token_prob.push_back(pi);

        float qi = sx.q_token_prob[static_cast<size_t>(i)];
        if (cfg.greedy_draft) {
            qi = 1.0f;
        }
        qi = std::max(qi, kEpsilon);

        float ratio = pi / qi;
        if (ratio > 1.0f) {
            ratio = 1.0f;
        }

        const float draw = std::uniform_real_distribution<float>(0.0f, 1.0f)(acceptance_rng);
        if (draw > ratio) {
            rejected = true;
            break;
        }

        if (!decode_single(sx.verifier_ctx, proposal)) {
            rejected = true;
            break;
        }

        ++accepted;
        if (repeat_window > 0) {
            push_repetition(verifier_window, proposal, repeat_window);
        }

        if (gptoss_vocab_is_eog(vocab, proposal)) {
            eos_token = proposal;
            break;
        }
    }

    if (!rejected && eos_token == GPTOSS_TOKEN_NULL && accepted == proposals_count && !encountered_eos) {
        const float * logits_raw = gptoss_get_logits(sx.verifier_ctx);
        if (logits_raw) {
            build_adjusted_probs(
                    logits_raw,
                    vocab_size,
                    temperature,
                    top_k,
                    top_p,
                    verifier_window,
                    repeat_penalty,
                    sx.p_next,
                    sx.candidate_order,
                    sx.candidate_scratch);
        }
    }

    if (!restore_sequence(sx.verifier_ctx, sx.verifier_state)) {
        return 0;
    }

    std::deque<gptoss_token> accept_window = base_window;
    for (int i = 0; i < accepted; ++i) {
        const gptoss_token tok = static_cast<gptoss_token>(sx.proposals[static_cast<size_t>(i)]);
        if (repeat_window > 0) {
            push_repetition(accept_window, tok, repeat_window);
        }
    }

    if (rejected && accepted < proposals_count) {
        if (!snapshot_sequence(sx.draft_ctx, sx.draft_state)) {
            return 0;
        }
        for (int i = 0; i < accepted; ++i) {
            const gptoss_token tok = static_cast<gptoss_token>(sx.proposals[static_cast<size_t>(i)]);
            if (!decode_single(sx.draft_ctx, tok)) {
                break;
            }
        }
        const float * logits_raw = gptoss_get_logits(sx.draft_ctx);
        if (logits_raw) {
            build_adjusted_probs(
                    logits_raw,
                    vocab_size,
                    temperature,
                    top_k,
                    top_p,
                    accept_window,
                    repeat_penalty,
                    sx.q_next,
                    sx.candidate_order,
                    sx.candidate_scratch);

            if (cfg.greedy_draft) {
                enforce_greedy_delta(sx.q_next, sx.candidate_order);
            }
        } else {
            std::fill(sx.q_next.begin(), sx.q_next.end(), 0.0f);
        }
        restore_sequence(sx.draft_ctx, sx.draft_state);

        float diff_sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            float value = sx.p_next[static_cast<size_t>(i)] - sx.q_next[static_cast<size_t>(i)];
            if (value < 0.0f) {
                value = 0.0f;
            }
            sx.q_next[static_cast<size_t>(i)] = value;
            diff_sum += value;
        }

        if (diff_sum > 0.0f) {
            for (int i = 0; i < vocab_size; ++i) {
                sx.p_next[static_cast<size_t>(i)] = sx.q_next[static_cast<size_t>(i)] / diff_sum;
            }
            sx.candidate_order.clear();
            for (int i = 0; i < vocab_size; ++i) {
                if (sx.p_next[static_cast<size_t>(i)] > 0.0f) {
                    sx.candidate_order.push_back(i);
                }
            }
            std::sort(
                    sx.candidate_order.begin(),
                    sx.candidate_order.end(),
                    [&](int a, int b) {
                        return sx.p_next[static_cast<size_t>(a)] > sx.p_next[static_cast<size_t>(b)];
                    });
        } else {
            sx.candidate_order.clear();
            for (int i = 0; i < vocab_size; ++i) {
                if (sx.p_next[static_cast<size_t>(i)] > 0.0f) {
                    sx.candidate_order.push_back(i);
                }
            }
            std::sort(
                    sx.candidate_order.begin(),
                    sx.candidate_order.end(),
                    [&](int a, int b) {
                        return sx.p_next[static_cast<size_t>(a)] > sx.p_next[static_cast<size_t>(b)];
                    });
        }
    }

    const bool eos_accepted = (eos_token != GPTOSS_TOKEN_NULL);

    gptoss_token correction = GPTOSS_TOKEN_NULL;
    bool emit_correction = !eos_accepted;
    if (emit_correction) {
        if (sx.candidate_order.empty()) {
            for (int i = 0; i < vocab_size; ++i) {
                if (sx.p_next[static_cast<size_t>(i)] > 0.0f) {
                    sx.candidate_order.push_back(i);
                }
            }
        }
        if (!sx.candidate_order.empty()) {
            correction = sample_from_probs(sx.p_next, sx.candidate_order, rng);
        } else {
            correction = static_cast<gptoss_token>(sx.proposals.empty() ? gptoss_vocab_eos(vocab) : sx.proposals.back());
        }
    }

    if (!restore_sequence(sx.draft_ctx, sx.draft_state)) {
        return 0;
    }
    if (!restore_sequence(sx.verifier_ctx, sx.verifier_state)) {
        return 0;
    }

    int appended = 0;
    for (int i = 0; i < accepted; ++i) {
        const gptoss_token tok = static_cast<gptoss_token>(sx.proposals[static_cast<size_t>(i)]);
        if (!decode_single(sx.verifier_ctx, tok)) {
            return appended;
        }
        if (!decode_single(sx.draft_ctx, tok)) {
            return appended;
        }
        out_tokens.push_back(tok);
        ++appended;
    }

    if (emit_correction) {
        if (!decode_single(sx.verifier_ctx, correction)) {
            return appended;
        }
        if (!decode_single(sx.draft_ctx, correction)) {
            return appended;
        }
        out_tokens.push_back(correction);
        ++appended;
    }

    mx.steps += 1;
    mx.proposed += static_cast<uint64_t>(proposals_count);
    mx.accepted += static_cast<uint64_t>(accepted);
    const double ratio = proposals_count > 0 ? static_cast<double>(accepted) / static_cast<double>(proposals_count) : 0.0;
    if (mx.steps == 1) {
        mx.ewma_accept = ratio;
    } else {
        const double alpha = 0.05;
        mx.ewma_accept = (1.0 - alpha) * mx.ewma_accept + alpha * ratio;
    }

    if (cfg.debug) {
        std::cerr << "[spec] step=" << mx.steps
                  << " proposed=" << proposals_count
                  << " accepted=" << accepted
                  << " appended=" << appended
                  << " ratio=" << ratio
                  << " ewma=" << mx.ewma_accept
                  << (emit_correction ? "" : " (no-correction)")
                  << std::endl;
    }

    return appended;
}

