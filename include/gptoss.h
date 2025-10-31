#ifndef GPTOSS_H
#define GPTOSS_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef GPTOSS_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GPTOSS_BUILD
#            define GPTOSS_API __declspec(dllexport)
#        else
#            define GPTOSS_API __declspec(dllimport)
#        endif
#    else
#        define GPTOSS_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GPTOSS_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define GPTOSS_DEFAULT_SEED 0xFFFFFFFF

#define GPTOSS_TOKEN_NULL -1

#define GPTOSS_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define GPTOSS_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define GPTOSS_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define GPTOSS_SESSION_MAGIC   GPTOSS_FILE_MAGIC_GGSN
#define GPTOSS_SESSION_VERSION 9

#define GPTOSS_STATE_SEQ_MAGIC   GPTOSS_FILE_MAGIC_GGSQ
#define GPTOSS_STATE_SEQ_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct gptoss_vocab;
    struct gptoss_model;
    struct gptoss_context;
    struct gptoss_sampler;

    typedef struct gptoss_memory_i * gptoss_memory_t;

    typedef int32_t gptoss_pos;
    typedef int32_t gptoss_token;
    typedef int32_t gptoss_seq_id;

    enum gptoss_vocab_type {
        GPTOSS_VOCAB_TYPE_NONE   = 0, // For models without vocab
        GPTOSS_VOCAB_TYPE_SPM    = 1, // GptOss tokenizer based on byte-level BPE with byte fallback
        GPTOSS_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        GPTOSS_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        GPTOSS_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        GPTOSS_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        GPTOSS_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

    enum gptoss_rope_type {
        GPTOSS_ROPE_TYPE_NONE   = -1,
        GPTOSS_ROPE_TYPE_NORM   = 0,
        GPTOSS_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        GPTOSS_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        GPTOSS_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum gptoss_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        GPTOSS_TOKEN_TYPE_UNDEFINED    = 0,
        GPTOSS_TOKEN_TYPE_NORMAL       = 1,
        GPTOSS_TOKEN_TYPE_UNKNOWN      = 2,
        GPTOSS_TOKEN_TYPE_CONTROL      = 3,
        GPTOSS_TOKEN_TYPE_USER_DEFINED = 4,
        GPTOSS_TOKEN_TYPE_UNUSED       = 5,
        GPTOSS_TOKEN_TYPE_BYTE         = 6,
    };

    enum gptoss_token_attr {
        GPTOSS_TOKEN_ATTR_UNDEFINED    = 0,
        GPTOSS_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        GPTOSS_TOKEN_ATTR_UNUSED       = 1 << 1,
        GPTOSS_TOKEN_ATTR_NORMAL       = 1 << 2,
        GPTOSS_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        GPTOSS_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        GPTOSS_TOKEN_ATTR_BYTE         = 1 << 5,
        GPTOSS_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        GPTOSS_TOKEN_ATTR_LSTRIP       = 1 << 7,
        GPTOSS_TOKEN_ATTR_RSTRIP       = 1 << 8,
        GPTOSS_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum gptoss_ftype {
        GPTOSS_FTYPE_ALL_F32              = 0,
        GPTOSS_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // GPTOSS_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // GPTOSS_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // GPTOSS_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        GPTOSS_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //GPTOSS_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //GPTOSS_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //GPTOSS_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        GPTOSS_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
        GPTOSS_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors

        GPTOSS_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum gptoss_rope_scaling_type {
        GPTOSS_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        GPTOSS_ROPE_SCALING_TYPE_NONE        = 0,
        GPTOSS_ROPE_SCALING_TYPE_LINEAR      = 1,
        GPTOSS_ROPE_SCALING_TYPE_YARN        = 2,
        GPTOSS_ROPE_SCALING_TYPE_LONGROPE    = 3,
        GPTOSS_ROPE_SCALING_TYPE_MAX_VALUE   = GPTOSS_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum gptoss_pooling_type {
        GPTOSS_POOLING_TYPE_UNSPECIFIED = -1,
        GPTOSS_POOLING_TYPE_NONE = 0,
        GPTOSS_POOLING_TYPE_MEAN = 1,
        GPTOSS_POOLING_TYPE_CLS  = 2,
        GPTOSS_POOLING_TYPE_LAST = 3,
        GPTOSS_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum gptoss_attention_type {
        GPTOSS_ATTENTION_TYPE_UNSPECIFIED = -1,
        GPTOSS_ATTENTION_TYPE_CAUSAL      = 0,
        GPTOSS_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum gptoss_flash_attn_type {
        GPTOSS_FLASH_ATTN_TYPE_AUTO     = -1,
        GPTOSS_FLASH_ATTN_TYPE_DISABLED = 0,
        GPTOSS_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    GPTOSS_API const char * gptoss_flash_attn_type_name(enum gptoss_flash_attn_type flash_attn_type);

    enum gptoss_split_mode {
        GPTOSS_SPLIT_MODE_NONE  = 0, // single GPU
        GPTOSS_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
        GPTOSS_SPLIT_MODE_ROW   = 2, // split layers and KV across GPUs, use tensor parallelism if supported
    };

    // TODO: simplify (https://github.com/ggml-org/gptoss.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct gptoss_token_data {
        gptoss_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } gptoss_token_data;

    typedef struct gptoss_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        gptoss_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } gptoss_token_data_array;

    typedef bool (*gptoss_progress_callback)(float progress, void * user_data);

    // Input data for gptoss_encode/gptoss_decode
    // A gptoss_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by gptoss_encode/gptoss_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct gptoss_batch {
        int32_t n_tokens;

        gptoss_token  *  token;
        float        *  embd;
        gptoss_pos    *  pos;
        int32_t      *  n_seq_id;
        gptoss_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } gptoss_batch;

    enum gptoss_model_kv_override_type {
        GPTOSS_KV_OVERRIDE_TYPE_INT,
        GPTOSS_KV_OVERRIDE_TYPE_FLOAT,
        GPTOSS_KV_OVERRIDE_TYPE_BOOL,
        GPTOSS_KV_OVERRIDE_TYPE_STR,
    };

    struct gptoss_model_kv_override {
        enum gptoss_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct gptoss_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct gptoss_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct gptoss_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM
        enum gptoss_split_mode split_mode; // how to split the model across multiple GPUs

        // the GPU that is used for the entire model when split_mode is GPTOSS_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: gptoss_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        gptoss_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct gptoss_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;      // only load the vocabulary, no weights
        bool use_mmap;        // use mmap if possible
        bool use_mlock;       // force system to keep model in RAM
        bool check_tensors;   // validate model tensor data
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer allowing extra buffers to be used
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggml-org/gptoss.cpp/pull/7544
    struct gptoss_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to gptoss_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum gptoss_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum gptoss_rope_scaling_type`
        enum gptoss_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum gptoss_attention_type    attention_type;    // attention type to use for embeddings
        enum gptoss_flash_attn_type   flash_attn_type;   // when to enable Flash Attention

        // ref: https://github.com/ggml-org/gptoss.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        // Abort callback
        // if it returns true, execution of gptoss_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/gptoss.cpp/pull/13194#issuecomment-2868343055)
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          //       ref: https://github.com/ggml-org/gptoss.cpp/pull/13845#issuecomment-2924800573
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          // ref: https://github.com/ggml-org/gptoss.cpp/pull/14363
    };

    // model quantization parameters
    typedef struct gptoss_model_quantize_params {
        int32_t nthread;                      // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum gptoss_ftype ftype;               // quantize to this gptoss_ftype
        enum ggml_type output_tensor_type;    // output tensor type
        enum ggml_type token_embedding_type;  // token embeddings tensor type
        bool allow_requantize;                // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;          // quantize output.weight
        bool only_copy;                       // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                            // quantize all tensors to the default type
        bool keep_split;                      // quantize to the same number of shards
        void * imatrix;                       // pointer to importance matrix data
        void * kv_overrides;                  // pointer to vector containing overrides
        void * tensor_types;                  // pointer to vector containing tensor types
        void * prune_layers;                  // pointer to vector containing layer indices to prune
    } gptoss_model_quantize_params;

    typedef struct gptoss_logit_bias {
        gptoss_token token;
        float bias;
    } gptoss_logit_bias;

    typedef struct gptoss_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } gptoss_sampler_chain_params;

    // used in chat template
    typedef struct gptoss_chat_message {
        const char * role;
        const char * content;
    } gptoss_chat_message;

    // lora adapter
    struct gptoss_adapter_lora;

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/gptoss.cpp/discussions/9172)
    GPTOSS_API struct gptoss_model_params          gptoss_model_default_params(void);
    GPTOSS_API struct gptoss_context_params        gptoss_context_default_params(void);
    GPTOSS_API struct gptoss_sampler_chain_params  gptoss_sampler_chain_default_params(void);
    GPTOSS_API struct gptoss_model_quantize_params gptoss_model_quantize_default_params(void);

    // Initialize the gptoss + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    GPTOSS_API void gptoss_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    GPTOSS_API void gptoss_backend_free(void);

    //optional:
    GPTOSS_API void gptoss_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    GPTOSS_API void gptoss_attach_threadpool(
            struct gptoss_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    GPTOSS_API void gptoss_detach_threadpool(struct gptoss_context * ctx);

    DEPRECATED(GPTOSS_API struct gptoss_model * gptoss_load_model_from_file(
                             const char * path_model,
              struct gptoss_model_params   params),
            "use gptoss_model_load_from_file instead");

    // Load the model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use gptoss_model_load_from_splits
    GPTOSS_API struct gptoss_model * gptoss_model_load_from_file(
                             const char * path_model,
              struct gptoss_model_params   params);

    // Load the model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    GPTOSS_API struct gptoss_model * gptoss_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct gptoss_model_params    params);

    GPTOSS_API void gptoss_model_save_to_file(
            const struct gptoss_model * model,
                        const char * path_model);

    DEPRECATED(GPTOSS_API void gptoss_free_model(struct gptoss_model * model),
            "use gptoss_model_free instead");

    GPTOSS_API void gptoss_model_free(struct gptoss_model * model);

    GPTOSS_API struct gptoss_context * gptoss_init_from_model(
                     struct gptoss_model * model,
            struct gptoss_context_params   params);

    DEPRECATED(GPTOSS_API struct gptoss_context * gptoss_new_context_with_model(
                     struct gptoss_model * model,
            struct gptoss_context_params   params),
            "use gptoss_init_from_model instead");

    // Frees all allocated memory
    GPTOSS_API void gptoss_free(struct gptoss_context * ctx);

    GPTOSS_API int64_t gptoss_time_us(void);

    GPTOSS_API size_t gptoss_max_devices(void);
    GPTOSS_API size_t gptoss_max_parallel_sequences(void);

    GPTOSS_API bool gptoss_supports_mmap       (void);
    GPTOSS_API bool gptoss_supports_mlock      (void);
    GPTOSS_API bool gptoss_supports_gpu_offload(void);
    GPTOSS_API bool gptoss_supports_rpc        (void);

    GPTOSS_API uint32_t gptoss_n_ctx      (const struct gptoss_context * ctx);
    GPTOSS_API uint32_t gptoss_n_batch    (const struct gptoss_context * ctx);
    GPTOSS_API uint32_t gptoss_n_ubatch   (const struct gptoss_context * ctx);
    GPTOSS_API uint32_t gptoss_n_seq_max  (const struct gptoss_context * ctx);

    DEPRECATED(GPTOSS_API int32_t gptoss_n_ctx_train(const struct gptoss_model * model), "use gptoss_model_n_ctx_train instead");
    DEPRECATED(GPTOSS_API int32_t gptoss_n_embd     (const struct gptoss_model * model), "use gptoss_model_n_embd instead");
    DEPRECATED(GPTOSS_API int32_t gptoss_n_layer    (const struct gptoss_model * model), "use gptoss_model_n_layer instead");
    DEPRECATED(GPTOSS_API int32_t gptoss_n_head     (const struct gptoss_model * model), "use gptoss_model_n_head instead");

    DEPRECATED(GPTOSS_API int32_t gptoss_n_vocab    (const struct gptoss_vocab * vocab), "use gptoss_vocab_n_tokens instead");

    GPTOSS_API const struct gptoss_model * gptoss_get_model   (const struct gptoss_context * ctx);
    GPTOSS_API           gptoss_memory_t   gptoss_get_memory  (const struct gptoss_context * ctx);
    GPTOSS_API  enum gptoss_pooling_type   gptoss_pooling_type(const struct gptoss_context * ctx); // TODO: rename to gptoss_get_pooling_type

    GPTOSS_API const struct gptoss_vocab * gptoss_model_get_vocab(const struct gptoss_model * model);
    GPTOSS_API enum gptoss_rope_type       gptoss_model_rope_type(const struct gptoss_model * model);

    GPTOSS_API int32_t gptoss_model_n_ctx_train(const struct gptoss_model * model);
    GPTOSS_API int32_t gptoss_model_n_embd     (const struct gptoss_model * model);
    GPTOSS_API int32_t gptoss_model_n_layer    (const struct gptoss_model * model);
    GPTOSS_API int32_t gptoss_model_n_head     (const struct gptoss_model * model);
    GPTOSS_API int32_t gptoss_model_n_head_kv  (const struct gptoss_model * model);
    GPTOSS_API int32_t gptoss_model_n_swa      (const struct gptoss_model * model);

    // Get the model's RoPE frequency scaling factor
    GPTOSS_API float gptoss_model_rope_freq_scale_train(const struct gptoss_model * model);

    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    GPTOSS_API uint32_t gptoss_model_n_cls_out(const struct gptoss_model * model);

    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    GPTOSS_API const char * gptoss_model_cls_label(const struct gptoss_model * model, uint32_t i);

    GPTOSS_API enum gptoss_vocab_type gptoss_vocab_type(const struct gptoss_vocab * vocab);

    GPTOSS_API int32_t gptoss_vocab_n_tokens(const struct gptoss_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    GPTOSS_API int32_t gptoss_model_meta_val_str(const struct gptoss_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    GPTOSS_API int32_t gptoss_model_meta_count(const struct gptoss_model * model);

    // Get metadata key name by index
    GPTOSS_API int32_t gptoss_model_meta_key_by_index(const struct gptoss_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    GPTOSS_API int32_t gptoss_model_meta_val_str_by_index(const struct gptoss_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    GPTOSS_API int32_t gptoss_model_desc(const struct gptoss_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    GPTOSS_API uint64_t gptoss_model_size(const struct gptoss_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    GPTOSS_API const char * gptoss_model_chat_template(const struct gptoss_model * model, const char * name);

    // Returns the total number of parameters in the model
    GPTOSS_API uint64_t gptoss_model_n_params(const struct gptoss_model * model);

    // Returns true if the model contains an encoder that requires gptoss_encode() call
    GPTOSS_API bool gptoss_model_has_encoder(const struct gptoss_model * model);

    // Returns true if the model contains a decoder that requires gptoss_decode() call
    GPTOSS_API bool gptoss_model_has_decoder(const struct gptoss_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    GPTOSS_API gptoss_token gptoss_model_decoder_start_token(const struct gptoss_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    GPTOSS_API bool gptoss_model_is_recurrent(const struct gptoss_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    GPTOSS_API bool gptoss_model_is_hybrid(const struct gptoss_model * model);

    // Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    GPTOSS_API bool gptoss_model_is_diffusion(const struct gptoss_model * model);

    // Returns 0 on success
    GPTOSS_API uint32_t gptoss_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const gptoss_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    GPTOSS_API struct gptoss_adapter_lora * gptoss_adapter_lora_init(
            struct gptoss_model * model,
            const char * path_lora);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    GPTOSS_API int32_t gptoss_adapter_meta_val_str(const struct gptoss_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    GPTOSS_API int32_t gptoss_adapter_meta_count(const struct gptoss_adapter_lora * adapter);

    // Get metadata key name by index
    GPTOSS_API int32_t gptoss_adapter_meta_key_by_index(const struct gptoss_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    GPTOSS_API int32_t gptoss_adapter_meta_val_str_by_index(const struct gptoss_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter
    // Note: loaded adapters will be free when the associated model is deleted
    GPTOSS_API void gptoss_adapter_lora_free(struct gptoss_adapter_lora * adapter);

    // Get the invocation tokens if the current lora is an alora
    GPTOSS_API uint64_t            gptoss_adapter_get_alora_n_invocation_tokens(const struct gptoss_adapter_lora * adapter);
    GPTOSS_API const gptoss_token * gptoss_adapter_get_alora_invocation_tokens  (const struct gptoss_adapter_lora * adapter);

    // The following functions operate on a gptoss_context, hence the naming: gptoss_verb_...

    // Add a loaded LoRA adapter to given context
    // This will not modify model's weight
    GPTOSS_API int32_t gptoss_set_adapter_lora(
            struct gptoss_context * ctx,
            struct gptoss_adapter_lora * adapter,
            float scale);

    // Remove a specific LoRA adapter from given context
    // Return -1 if the adapter is not present in the context
    GPTOSS_API int32_t gptoss_rm_adapter_lora(
            struct gptoss_context * ctx,
            struct gptoss_adapter_lora * adapter);

    // Remove all LoRA adapters from given context
    GPTOSS_API void gptoss_clear_adapter_lora(struct gptoss_context * ctx);

    // Apply a loaded control vector to a gptoss_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See gptoss_control_vector_load in common to load a control vector.
    GPTOSS_API int32_t gptoss_apply_adapter_cvec(
            struct gptoss_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    GPTOSS_API void gptoss_memory_clear(
            gptoss_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    GPTOSS_API bool gptoss_memory_seq_rm(
            gptoss_memory_t mem,
              gptoss_seq_id seq_id,
                 gptoss_pos p0,
                 gptoss_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    GPTOSS_API void gptoss_memory_seq_cp(
            gptoss_memory_t mem,
              gptoss_seq_id seq_id_src,
              gptoss_seq_id seq_id_dst,
                 gptoss_pos p0,
                 gptoss_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    GPTOSS_API void gptoss_memory_seq_keep(
            gptoss_memory_t mem,
              gptoss_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    GPTOSS_API void gptoss_memory_seq_add(
            gptoss_memory_t mem,
              gptoss_seq_id seq_id,
                 gptoss_pos p0,
                 gptoss_pos p1,
                 gptoss_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    GPTOSS_API void gptoss_memory_seq_div(
            gptoss_memory_t mem,
              gptoss_seq_id seq_id,
                 gptoss_pos p0,
                 gptoss_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    GPTOSS_API gptoss_pos gptoss_memory_seq_pos_min(
            gptoss_memory_t mem,
              gptoss_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    GPTOSS_API gptoss_pos gptoss_memory_seq_pos_max(
            gptoss_memory_t mem,
              gptoss_seq_id seq_id);

    // Check if the memory supports shifting
    GPTOSS_API bool gptoss_memory_can_shift(gptoss_memory_t mem);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    GPTOSS_API size_t gptoss_state_get_size(struct gptoss_context * ctx);
    GPTOSS_API DEPRECATED(size_t gptoss_get_state_size(struct gptoss_context * ctx),
        "use gptoss_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    GPTOSS_API size_t gptoss_state_get_data(
            struct gptoss_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    GPTOSS_API DEPRECATED(size_t gptoss_copy_state_data(
            struct gptoss_context * ctx,
                         uint8_t * dst),
        "use gptoss_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    GPTOSS_API size_t gptoss_state_set_data(
            struct gptoss_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    GPTOSS_API DEPRECATED(size_t gptoss_set_state_data(
            struct gptoss_context * ctx,
                   const uint8_t * src),
        "use gptoss_state_set_data instead");

    // Save/load session file
    GPTOSS_API bool gptoss_state_load_file(
            struct gptoss_context * ctx,
                      const char * path_session,
                     gptoss_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    GPTOSS_API DEPRECATED(bool gptoss_load_session_file(
            struct gptoss_context * ctx,
                      const char * path_session,
                     gptoss_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use gptoss_state_load_file instead");

    GPTOSS_API bool gptoss_state_save_file(
            struct gptoss_context * ctx,
                      const char * path_session,
               const gptoss_token * tokens,
                          size_t   n_token_count);
    GPTOSS_API DEPRECATED(bool gptoss_save_session_file(
            struct gptoss_context * ctx,
                      const char * path_session,
               const gptoss_token * tokens,
                          size_t   n_token_count),
        "use gptoss_state_save_file instead");

    // Get the exact size needed to copy the state of a single sequence
    GPTOSS_API size_t gptoss_state_seq_get_size(
            struct gptoss_context * ctx,
                    gptoss_seq_id   seq_id);

    // Copy the state of a single sequence into the specified buffer
    GPTOSS_API size_t gptoss_state_seq_get_data(
            struct gptoss_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    gptoss_seq_id   seq_id);

    // Copy the sequence data (originally copied with `gptoss_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    GPTOSS_API size_t gptoss_state_seq_set_data(
            struct gptoss_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    gptoss_seq_id   dest_seq_id);

    GPTOSS_API size_t gptoss_state_seq_save_file(
            struct gptoss_context * ctx,
                      const char * filepath,
                    gptoss_seq_id   seq_id,
               const gptoss_token * tokens,
                          size_t   n_token_count);

    GPTOSS_API size_t gptoss_state_seq_load_file(
            struct gptoss_context * ctx,
                      const char * filepath,
                    gptoss_seq_id   dest_seq_id,
                     gptoss_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

// for backwards-compat
#define GPTOSS_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define GPTOSS_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

    typedef uint32_t gptoss_state_seq_flags;

    GPTOSS_API size_t gptoss_state_seq_get_size_ext(
            struct gptoss_context * ctx,
                    gptoss_seq_id   seq_id,
           gptoss_state_seq_flags   flags);

    GPTOSS_API size_t gptoss_state_seq_get_data_ext(
            struct gptoss_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    gptoss_seq_id   seq_id,
           gptoss_state_seq_flags   flags);

    GPTOSS_API size_t gptoss_state_seq_set_data_ext(
            struct gptoss_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    gptoss_seq_id   dest_seq_id,
           gptoss_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by gptoss_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    GPTOSS_API struct gptoss_batch gptoss_batch_get_one(
                  gptoss_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with gptoss_batch_free()
    // If embd != 0, gptoss_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, gptoss_batch.token will be allocated to store n_tokens gptoss_token
    // The rest of the gptoss_batch members are allocated with size n_tokens
    // All members are left uninitialized
    GPTOSS_API struct gptoss_batch gptoss_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with gptoss_batch_init()
    GPTOSS_API void gptoss_batch_free(struct gptoss_batch batch);

    // Process a batch of tokens.
    // In contrast to gptoss_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    GPTOSS_API int32_t gptoss_encode(
            struct gptoss_context * ctx,
              struct gptoss_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using gptoss_memory_seq_pos_min() and gptoss_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    GPTOSS_API int32_t gptoss_decode(
            struct gptoss_context * ctx,
              struct gptoss_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    GPTOSS_API void gptoss_set_n_threads(struct gptoss_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    GPTOSS_API int32_t gptoss_n_threads(struct gptoss_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    GPTOSS_API int32_t gptoss_n_threads_batch(struct gptoss_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with gptoss_get_embeddings()
    GPTOSS_API void gptoss_set_embeddings(struct gptoss_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    GPTOSS_API void gptoss_set_causal_attn(struct gptoss_context * ctx, bool causal_attn);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during gptoss_decode() to load and cache their weights.
    GPTOSS_API void gptoss_set_warmup(struct gptoss_context * ctx, bool warmup);

    // Set abort callback
    GPTOSS_API void gptoss_set_abort_callback(struct gptoss_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    GPTOSS_API void gptoss_synchronize(struct gptoss_context * ctx);

    // Token logits obtained from the last call to gptoss_decode()
    // The logits for which gptoss_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which gptoss_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of gptoss_get_logits_ith() (ref: https://github.com/ggml-org/gptoss.cpp/pull/14853#issuecomment-3113143522)
    GPTOSS_API float * gptoss_get_logits(struct gptoss_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // gptoss_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    GPTOSS_API float * gptoss_get_logits_ith(struct gptoss_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == GPTOSS_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which gptoss_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of gptoss_get_embeddings_ith() (ref: https://github.com/ggml-org/gptoss.cpp/pull/14853#issuecomment-3113143522)
    GPTOSS_API float * gptoss_get_embeddings(struct gptoss_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // gptoss_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    GPTOSS_API float * gptoss_get_embeddings_ith(struct gptoss_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is GPTOSS_POOLING_TYPE_NONE
    // when pooling_type == GPTOSS_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    GPTOSS_API float * gptoss_get_embeddings_seq(struct gptoss_context * ctx, gptoss_seq_id seq_id);

    //
    // Vocab
    //

    GPTOSS_API const char * gptoss_vocab_get_text(const struct gptoss_vocab * vocab, gptoss_token token);

    GPTOSS_API float gptoss_vocab_get_score(const struct gptoss_vocab * vocab, gptoss_token token);

    GPTOSS_API enum gptoss_token_attr gptoss_vocab_get_attr(const struct gptoss_vocab * vocab, gptoss_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    GPTOSS_API bool gptoss_vocab_is_eog(const struct gptoss_vocab * vocab, gptoss_token token);

    // Identify if Token Id is a control token or a render-able token
    GPTOSS_API bool gptoss_vocab_is_control(const struct gptoss_vocab * vocab, gptoss_token token);

    // Special tokens
    GPTOSS_API gptoss_token gptoss_vocab_bos(const struct gptoss_vocab * vocab); // beginning-of-sentence
    GPTOSS_API gptoss_token gptoss_vocab_eos(const struct gptoss_vocab * vocab); // end-of-sentence
    GPTOSS_API gptoss_token gptoss_vocab_eot(const struct gptoss_vocab * vocab); // end-of-turn
    GPTOSS_API gptoss_token gptoss_vocab_sep(const struct gptoss_vocab * vocab); // sentence separator
    GPTOSS_API gptoss_token gptoss_vocab_nl (const struct gptoss_vocab * vocab); // next-line
    GPTOSS_API gptoss_token gptoss_vocab_pad(const struct gptoss_vocab * vocab); // padding
    GPTOSS_API gptoss_token gptoss_vocab_mask(const struct gptoss_vocab * vocab); // mask

    GPTOSS_API bool gptoss_vocab_get_add_bos(const struct gptoss_vocab * vocab);
    GPTOSS_API bool gptoss_vocab_get_add_eos(const struct gptoss_vocab * vocab);
    GPTOSS_API bool gptoss_vocab_get_add_sep(const struct gptoss_vocab * vocab);

    GPTOSS_API gptoss_token gptoss_vocab_fim_pre(const struct gptoss_vocab * vocab);
    GPTOSS_API gptoss_token gptoss_vocab_fim_suf(const struct gptoss_vocab * vocab);
    GPTOSS_API gptoss_token gptoss_vocab_fim_mid(const struct gptoss_vocab * vocab);
    GPTOSS_API gptoss_token gptoss_vocab_fim_pad(const struct gptoss_vocab * vocab);
    GPTOSS_API gptoss_token gptoss_vocab_fim_rep(const struct gptoss_vocab * vocab);
    GPTOSS_API gptoss_token gptoss_vocab_fim_sep(const struct gptoss_vocab * vocab);

    DEPRECATED(GPTOSS_API const char * gptoss_token_get_text(const struct gptoss_vocab * vocab, gptoss_token token), "use gptoss_vocab_get_text instead");
    DEPRECATED(GPTOSS_API float gptoss_token_get_score(const struct gptoss_vocab * vocab, gptoss_token token), "use gptoss_vocab_get_score instead");
    DEPRECATED(GPTOSS_API enum gptoss_token_attr gptoss_token_get_attr(const struct gptoss_vocab * vocab, gptoss_token token), "use gptoss_vocab_get_attr instead");
    DEPRECATED(GPTOSS_API bool gptoss_token_is_eog(const struct gptoss_vocab * vocab, gptoss_token token), "use gptoss_vocab_is_eog instead");
    DEPRECATED(GPTOSS_API bool gptoss_token_is_control(const struct gptoss_vocab * vocab, gptoss_token token), "use gptoss_vocab_is_control instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_bos(const struct gptoss_vocab * vocab), "use gptoss_vocab_bos instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_eos(const struct gptoss_vocab * vocab), "use gptoss_vocab_eos instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_eot(const struct gptoss_vocab * vocab), "use gptoss_vocab_eot instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_cls(const struct gptoss_vocab * vocab), "use gptoss_vocab_cls instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_sep(const struct gptoss_vocab * vocab), "use gptoss_vocab_sep instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_nl (const struct gptoss_vocab * vocab), "use gptoss_vocab_nl instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_pad(const struct gptoss_vocab * vocab), "use gptoss_vocab_pad instead");
    DEPRECATED(GPTOSS_API bool gptoss_add_bos_token(const struct gptoss_vocab * vocab), "use gptoss_vocab_get_add_bos instead");
    DEPRECATED(GPTOSS_API bool gptoss_add_eos_token(const struct gptoss_vocab * vocab), "use gptoss_vocab_get_add_eos instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_fim_pre(const struct gptoss_vocab * vocab), "use gptoss_vocab_fim_pre instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_fim_suf(const struct gptoss_vocab * vocab), "use gptoss_vocab_fim_suf instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_fim_mid(const struct gptoss_vocab * vocab), "use gptoss_vocab_fim_mid instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_fim_pad(const struct gptoss_vocab * vocab), "use gptoss_vocab_fim_pad instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_fim_rep(const struct gptoss_vocab * vocab), "use gptoss_vocab_fim_rep instead");
    DEPRECATED(GPTOSS_API gptoss_token gptoss_token_fim_sep(const struct gptoss_vocab * vocab), "use gptoss_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(GPTOSS_API gptoss_token gptoss_vocab_cls(const struct gptoss_vocab * vocab), // classification
            "use gptoss_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    GPTOSS_API int32_t gptoss_tokenize(
        const struct gptoss_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     gptoss_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    GPTOSS_API int32_t gptoss_token_to_piece(
              const struct gptoss_vocab * vocab,
                           gptoss_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of gptoss_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    GPTOSS_API int32_t gptoss_detokenize(
        const struct gptoss_vocab * vocab,
               const gptoss_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    /// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/gptoss.cpp/wiki/Templates-supported-by-gptoss_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    /// @param chat Pointer to a list of multiple gptoss_chat_message
    /// @param n_msg Number of gptoss_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    GPTOSS_API int32_t gptoss_chat_apply_template(
                            const char * tmpl,
       const struct gptoss_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    GPTOSS_API int32_t gptoss_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = gptoss_sampler_chain_default_params();
    //
    //    gptoss_sampler * smpl = gptoss_sampler_chain_init(sparams);
    //
    //    gptoss_sampler_chain_add(smpl, gptoss_sampler_init_top_k(50));
    //    gptoss_sampler_chain_add(smpl, gptoss_sampler_init_top_p(0.9, 1));
    //    gptoss_sampler_chain_add(smpl, gptoss_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    gptoss_sampler_chain_add(smpl, gptoss_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        gptoss_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const gptoss_token id = gptoss_sampler_sample(smpl, ctx, -1);
    //
    //        // accepting the token updates the internal state of certain samplers (e.g. grammar, repetition, etc.)
    //        gptoss_sampler_accept(smpl, id);
    //        ...
    //    }
    //
    //    gptoss_sampler_free(smpl);
    //
    // TODO: In the future, gptoss_sampler will be utilized to offload the sampling to the backends (e.g. GPU).
    //

    typedef void * gptoss_sampler_context_t;

    // user code can implement the interface below in order to create custom gptoss_sampler
    struct gptoss_sampler_i {
        const char *           (*name)  (const struct gptoss_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct gptoss_sampler * smpl, gptoss_token token);              // can be NULL
        void                   (*apply) (      struct gptoss_sampler * smpl, gptoss_token_data_array * cur_p); // required
        void                   (*reset) (      struct gptoss_sampler * smpl);                                 // can be NULL
        struct gptoss_sampler * (*clone) (const struct gptoss_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct gptoss_sampler * smpl);                                 // can be NULL if ctx is NULL

        // TODO: API for internal libgptoss usage for appending the sampling to an existing ggml_cgraph
        //void (*apply_ggml) (struct gptoss_sampler * smpl, ...);
    };

    struct gptoss_sampler {
        const struct gptoss_sampler_i * iface;
        gptoss_sampler_context_t        ctx;
    };

    // mirror of gptoss_sampler_i:
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init  (const struct gptoss_sampler_i * iface, gptoss_sampler_context_t ctx);
    GPTOSS_API const char *           gptoss_sampler_name  (const struct gptoss_sampler * smpl);
    GPTOSS_API void                   gptoss_sampler_accept(      struct gptoss_sampler * smpl, gptoss_token token);
    GPTOSS_API void                   gptoss_sampler_apply (      struct gptoss_sampler * smpl, gptoss_token_data_array * cur_p);
    GPTOSS_API void                   gptoss_sampler_reset (      struct gptoss_sampler * smpl);
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_clone (const struct gptoss_sampler * smpl);
    // important: do not free if the sampler has been added to a gptoss_sampler_chain (via gptoss_sampler_chain_add)
    GPTOSS_API void                   gptoss_sampler_free  (      struct gptoss_sampler * smpl);

    // gptoss_sampler_chain
    // a type of gptoss_sampler that can chain multiple samplers one after another

    GPTOSS_API struct gptoss_sampler * gptoss_sampler_chain_init(struct gptoss_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when gptoss_sampler_free is called
    GPTOSS_API void                   gptoss_sampler_chain_add(      struct gptoss_sampler * chain, struct gptoss_sampler * smpl);
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_chain_get(const struct gptoss_sampler * chain, int32_t i);
    GPTOSS_API int                    gptoss_sampler_chain_n  (const struct gptoss_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_chain_remove(   struct gptoss_sampler * chain, int32_t i);

    // available samplers:

    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_greedy(void);
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_dist  (uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/gptoss.cpp/pull/3841
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `gptoss_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `gptoss_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Intializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_grammar(
            const struct gptoss_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_grammar_lazy(
            const struct gptoss_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const gptoss_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use gptoss_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/gptoss.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_grammar_lazy_patterns(
        const struct gptoss_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const gptoss_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_dry(
            const struct gptoss_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const gptoss_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    GPTOSS_API struct gptoss_sampler * gptoss_sampler_init_infill(const struct gptoss_vocab * vocab);

    // Returns the seed used by the sampler if applicable, GPTOSS_DEFAULT_SEED otherwise
    GPTOSS_API uint32_t gptoss_sampler_get_seed(const struct gptoss_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = gptoss_get_logits_ith(ctx, idx);
    //    gptoss_token_data_array cur_p = { ... init from logits ... };
    //    gptoss_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    gptoss_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    GPTOSS_API gptoss_token gptoss_sampler_sample(struct gptoss_sampler * smpl, struct gptoss_context * ctx, int32_t idx);

    // TODO: extend in the future
    //GPTOSS_API void gptoss_decode_with_sampler(struct gptoss_context * ctx, struct gptoss_sampler * smpl, struct gptoss_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          gptoss_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    GPTOSS_API int gptoss_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          gptoss_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    GPTOSS_API int gptoss_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

    // Print system information
    GPTOSS_API const char * gptoss_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    GPTOSS_API void gptoss_log_set(ggml_log_callback log_callback, void * user_data);

    //
    // Performance utils
    //
    // NOTE: Used by gptoss.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct gptoss_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
    };

    struct gptoss_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    GPTOSS_API struct gptoss_perf_context_data gptoss_perf_context      (const struct gptoss_context * ctx);
    GPTOSS_API void                           gptoss_perf_context_print(const struct gptoss_context * ctx);
    GPTOSS_API void                           gptoss_perf_context_reset(      struct gptoss_context * ctx);

    // NOTE: the following work only with samplers constructed via gptoss_sampler_chain_init
    GPTOSS_API struct gptoss_perf_sampler_data gptoss_perf_sampler      (const struct gptoss_sampler * chain);
    GPTOSS_API void                           gptoss_perf_sampler_print(const struct gptoss_sampler * chain);
    GPTOSS_API void                           gptoss_perf_sampler_reset(      struct gptoss_sampler * chain);

    // print a breakdown of per-device memory use via GPTOSS_LOG:
    GPTOSS_API void gptoss_memory_breakdown_print(const struct gptoss_context * ctx);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*gptoss_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    GPTOSS_API bool gptoss_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct gptoss_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in gptoss_context if 0

        gptoss_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    GPTOSS_API void gptoss_opt_init(struct gptoss_context * lctx, struct gptoss_model * model, struct gptoss_opt_params lopt_params);

    GPTOSS_API void gptoss_opt_epoch(
            struct gptoss_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#endif // GPTOSS_H
