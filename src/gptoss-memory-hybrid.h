#pragma once

#include "gptoss-batch.h"
#include "gptoss-graph.h"
#include "gptoss-kv-cache.h"
#include "gptoss-memory.h"
#include "gptoss-memory-recurrent.h"

#include <memory>
#include <vector>

//
// gptoss_memory_hybrid
//

// utilizes instances of gptoss_memory_recurrent and gptoss_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class gptoss_memory_hybrid : public gptoss_memory_i {
public:
    gptoss_memory_hybrid(
        const gptoss_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           gptoss_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    ~gptoss_memory_hybrid() = default;

    //
    // gptoss_memory_i
    //

    gptoss_memory_context_ptr init_batch(
            gptoss_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    gptoss_memory_context_ptr init_full() override;

    gptoss_memory_context_ptr init_update(gptoss_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (gptoss_seq_id seq_id,                              gptoss_pos p0, gptoss_pos p1) override;
    void seq_cp  (gptoss_seq_id seq_id_src, gptoss_seq_id seq_id_dst, gptoss_pos p0, gptoss_pos p1) override;
    void seq_keep(gptoss_seq_id seq_id)                                                          override;
    void seq_add (gptoss_seq_id seq_id,                              gptoss_pos p0, gptoss_pos p1, gptoss_pos shift) override;
    void seq_div (gptoss_seq_id seq_id,                              gptoss_pos p0, gptoss_pos p1, int d) override;

    gptoss_pos seq_pos_min(gptoss_seq_id seq_id) const override;
    gptoss_pos seq_pos_max(gptoss_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(gptoss_io_write_i & io, gptoss_seq_id seq_id = -1, gptoss_state_seq_flags flags = 0) const override;
    void state_read (gptoss_io_read_i  & io, gptoss_seq_id seq_id = -1, gptoss_state_seq_flags flags = 0)       override;

    //
    // gptoss_memory_hybrid specific API
    //

    gptoss_kv_cache * get_mem_attn() const;
    gptoss_memory_recurrent * get_mem_recr() const;

private:
    const gptoss_hparams & hparams;

    const std::unique_ptr<gptoss_kv_cache> mem_attn;
    const std::unique_ptr<gptoss_memory_recurrent> mem_recr;
};

class gptoss_memory_hybrid_context : public gptoss_memory_context_i {
public:
    using slot_info_vec_t = gptoss_kv_cache::slot_info_vec_t;

    // init failure
    explicit gptoss_memory_hybrid_context(gptoss_memory_status status);

    // init full
    explicit gptoss_memory_hybrid_context(gptoss_memory_hybrid * mem);

    // init update
    explicit gptoss_memory_hybrid_context(
        gptoss_memory_hybrid * mem,
              gptoss_context * lctx,
                       bool   optimize);

    // init success
    gptoss_memory_hybrid_context(
              gptoss_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<gptoss_ubatch>   ubatches);

    ~gptoss_memory_hybrid_context() = default;

    bool next()  override;
    bool apply() override;

    gptoss_memory_status  get_status() const override;
    const gptoss_ubatch & get_ubatch() const override;

    //
    // gptoss_memory_hybrid_context
    //

    const gptoss_kv_cache_context * get_attn() const;
    const gptoss_memory_recurrent_context * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<gptoss_ubatch> ubatches;

    const gptoss_memory_context_ptr ctx_attn;
    const gptoss_memory_context_ptr ctx_recr;

    const gptoss_memory_status status;
};
