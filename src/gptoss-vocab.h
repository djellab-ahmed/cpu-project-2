#pragma once

#include "gptoss.h"

#include <string>
#include <vector>
#include <memory>

// pre-tokenization types
enum gptoss_vocab_pre_type {
    GPTOSS_VOCAB_PRE_TYPE_DEFAULT         = 0,
    GPTOSS_VOCAB_PRE_TYPE_GPTOSS3          = 1,
    GPTOSS_VOCAB_PRE_TYPE_DEEPSEEK_LLM    = 2,
    GPTOSS_VOCAB_PRE_TYPE_DEEPSEEK_CODER  = 3,
    GPTOSS_VOCAB_PRE_TYPE_FALCON          = 4,
    GPTOSS_VOCAB_PRE_TYPE_MPT             = 5,
    GPTOSS_VOCAB_PRE_TYPE_STARCODER       = 6,
    GPTOSS_VOCAB_PRE_TYPE_GPT2            = 7,
    GPTOSS_VOCAB_PRE_TYPE_REFACT          = 8,
    GPTOSS_VOCAB_PRE_TYPE_COMMAND_R       = 9,
    GPTOSS_VOCAB_PRE_TYPE_STABLELM2       = 10,
    GPTOSS_VOCAB_PRE_TYPE_QWEN2           = 11,
    GPTOSS_VOCAB_PRE_TYPE_OLMO            = 12,
    GPTOSS_VOCAB_PRE_TYPE_DBRX            = 13,
    GPTOSS_VOCAB_PRE_TYPE_SMAUG           = 14,
    GPTOSS_VOCAB_PRE_TYPE_PORO            = 15,
    GPTOSS_VOCAB_PRE_TYPE_CHATGLM3        = 16,
    GPTOSS_VOCAB_PRE_TYPE_CHATGLM4        = 17,
    GPTOSS_VOCAB_PRE_TYPE_VIKING          = 18,
    GPTOSS_VOCAB_PRE_TYPE_JAIS            = 19,
    GPTOSS_VOCAB_PRE_TYPE_TEKKEN          = 20,
    GPTOSS_VOCAB_PRE_TYPE_SMOLLM          = 21,
    GPTOSS_VOCAB_PRE_TYPE_CODESHELL       = 22,
    GPTOSS_VOCAB_PRE_TYPE_BLOOM           = 23,
    GPTOSS_VOCAB_PRE_TYPE_GPT3_FINNISH    = 24,
    GPTOSS_VOCAB_PRE_TYPE_EXAONE          = 25,
    GPTOSS_VOCAB_PRE_TYPE_CHAMELEON       = 26,
    GPTOSS_VOCAB_PRE_TYPE_MINERVA         = 27,
    GPTOSS_VOCAB_PRE_TYPE_DEEPSEEK3_LLM   = 28,
    GPTOSS_VOCAB_PRE_TYPE_GPT4O           = 29,
    GPTOSS_VOCAB_PRE_TYPE_SUPERBPE        = 30,
    GPTOSS_VOCAB_PRE_TYPE_TRILLION        = 31,
    GPTOSS_VOCAB_PRE_TYPE_BAILINGMOE      = 32,
    GPTOSS_VOCAB_PRE_TYPE_GPTOSS4          = 33,
    GPTOSS_VOCAB_PRE_TYPE_PIXTRAL         = 34,
    GPTOSS_VOCAB_PRE_TYPE_SEED_CODER      = 35,
    GPTOSS_VOCAB_PRE_TYPE_HUNYUAN         = 36,
    GPTOSS_VOCAB_PRE_TYPE_KIMI_K2         = 37,
    GPTOSS_VOCAB_PRE_TYPE_HUNYUAN_DENSE   = 38,
    GPTOSS_VOCAB_PRE_TYPE_GROK_2          = 39,
    GPTOSS_VOCAB_PRE_TYPE_GRANITE_DOCLING = 40,
};

struct LLM_KV;
struct gptoss_model_loader;

struct gptoss_vocab {
    struct token_data {
        std::string      text;
        float            score;
        gptoss_token_attr attr;
    };

    gptoss_vocab();
    ~gptoss_vocab();

    void load(gptoss_model_loader & ml, const LLM_KV & kv);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum gptoss_vocab_type     get_type()     const;
    enum gptoss_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (gptoss_token id) const;
    bool is_unknown     (gptoss_token id) const;
    bool is_control     (gptoss_token id) const;
    bool is_byte        (gptoss_token id) const;
    bool is_user_defined(gptoss_token id) const;
    bool is_unused      (gptoss_token id) const;
    bool is_eog         (gptoss_token id) const;

    uint8_t     token_to_byte(gptoss_token id) const;
    gptoss_token byte_to_token(uint8_t ch)     const;

    gptoss_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(gptoss_token id) const;

    const char *     token_get_text (gptoss_token id) const;
    float            token_get_score(gptoss_token id) const;
    gptoss_token_attr token_get_attr (gptoss_token id) const;

    gptoss_token token_bos() const;
    gptoss_token token_eos() const;
    gptoss_token token_eot() const;
    gptoss_token token_eom() const;
    gptoss_token token_unk() const;
    gptoss_token token_sep() const;
    gptoss_token token_nl () const;
    gptoss_token token_pad() const;
    gptoss_token token_mask() const;

    gptoss_token token_prefix() const;
    gptoss_token token_middle() const;
    gptoss_token token_suffix() const;

    gptoss_token token_fim_pre() const;
    gptoss_token token_fim_suf() const;
    gptoss_token token_fim_mid() const;
    gptoss_token token_fim_pad() const;
    gptoss_token token_fim_rep() const;
    gptoss_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  gptoss_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<gptoss_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  gptoss_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(gptoss_token token) const;

    int32_t detokenize(
            const gptoss_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<gptoss_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
