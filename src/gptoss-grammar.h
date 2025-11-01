#pragma once

#include "gptoss.h"

#include <map>
#include <regex>
#include <string>
#include <vector>

struct gptoss_vocab;

// grammar element type
enum gptoss_gretype {
    // end of rule definition
    GPTOSS_GRETYPE_END            = 0,

    // start of alternate definition for rule
    GPTOSS_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    GPTOSS_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    GPTOSS_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    GPTOSS_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding GPTOSS_GRETYPE_CHAR or GPTOSS_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    GPTOSS_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding GPTOSS_GRETYPE_CHAR or
    // GPTOSS_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    GPTOSS_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    GPTOSS_GRETYPE_CHAR_ANY       = 7,
};

typedef struct gptoss_grammar_element {
    enum gptoss_gretype type;
    uint32_t           value; // Unicode code point or rule ID
} gptoss_grammar_element;

struct gptoss_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct gptoss_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    gptoss_partial_utf8   partial_utf8;
};

using gptoss_grammar_rule  = std::vector<      gptoss_grammar_element>;
using gptoss_grammar_stack = std::vector<const gptoss_grammar_element *>;

using gptoss_grammar_rules      = std::vector<gptoss_grammar_rule>;
using gptoss_grammar_stacks     = std::vector<gptoss_grammar_stack>;
using gptoss_grammar_candidates = std::vector<gptoss_grammar_candidate>;

// TODO: remove, needed for tests atm
const gptoss_grammar_rules  & gptoss_grammar_get_rules (const struct gptoss_grammar * grammar);
      gptoss_grammar_stacks & gptoss_grammar_get_stacks(      struct gptoss_grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `gptoss_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void gptoss_grammar_accept(struct gptoss_grammar * grammar, uint32_t chr);

std::vector<gptoss_grammar_candidate> gptoss_grammar_reject_candidates_for_stack(
        const gptoss_grammar_rules      & rules,
        const gptoss_grammar_stack      & stack,
        const gptoss_grammar_candidates & candidates);

struct gptoss_grammar_parser {
    std::map<std::string, uint32_t> symbol_ids;

    gptoss_grammar_rules rules;

    gptoss_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const gptoss_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            gptoss_grammar_rule & rule,
            bool               is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct gptoss_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;
};

struct gptoss_grammar {
    // note: allow null vocab for testing (not great)
    const gptoss_vocab * vocab;

    const gptoss_grammar_rules  rules;  // TODO: shared ptr
          gptoss_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    gptoss_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<gptoss_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<gptoss_grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.

};

//
// internal API
//

// note: needed for tests (not great)
struct gptoss_grammar * gptoss_grammar_init_impl(
        const struct gptoss_vocab * vocab,
        const gptoss_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct gptoss_grammar * gptoss_grammar_init_impl(
        const struct gptoss_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const gptoss_token * trigger_tokens,
                            size_t num_trigger_tokens);

void gptoss_grammar_free_impl(struct gptoss_grammar * grammar);

struct gptoss_grammar * gptoss_grammar_clone_impl(const struct gptoss_grammar & grammar);

// TODO: move the API below as member functions of gptoss_grammar
void gptoss_grammar_apply_impl(
        const struct gptoss_grammar & grammar,
            gptoss_token_data_array * cur_p);

void gptoss_grammar_accept_impl(
              struct gptoss_grammar & grammar,
                       gptoss_token   token);

void gptoss_grammar_accept_str(
              struct gptoss_grammar & grammar,
                 const std::string & piece);
