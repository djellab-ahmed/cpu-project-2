#pragma once

#include "gptoss-kv-cache.h"

#include <vector>

//
// gptoss_kv_cache_iswa
//

// utilizes two instances of gptoss_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class gptoss_kv_cache_iswa : public gptoss_memory_i {
public:
    gptoss_kv_cache_iswa(
            const gptoss_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~gptoss_kv_cache_iswa() = default;

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
    void state_read (gptoss_io_read_i  & io, gptoss_seq_id seq_id = -1, gptoss_state_seq_flags flags = 0) override;

    //
    // gptoss_kv_cache_iswa specific API
    //

    gptoss_kv_cache * get_base() const;
    gptoss_kv_cache * get_swa () const;

private:
    const gptoss_hparams & hparams;

    const bool unified;

    std::unique_ptr<gptoss_kv_cache> kv_base;
    std::unique_ptr<gptoss_kv_cache> kv_swa;
};

class gptoss_kv_cache_iswa_context : public gptoss_memory_context_i {
public:
    using slot_info_vec_t = gptoss_kv_cache::slot_info_vec_t;

    // used for errors
    gptoss_kv_cache_iswa_context(gptoss_memory_status status);

    // used to create a full-cache context
    gptoss_kv_cache_iswa_context(
            gptoss_kv_cache_iswa * kv);

    // used to create an update context
    gptoss_kv_cache_iswa_context(
            gptoss_kv_cache_iswa * kv,
            gptoss_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    gptoss_kv_cache_iswa_context(
            gptoss_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<gptoss_ubatch> ubatches);

    virtual ~gptoss_kv_cache_iswa_context();

    //
    // gptoss_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    gptoss_memory_status  get_status() const override;
    const gptoss_ubatch & get_ubatch() const override;

    //
    // gptoss_kv_cache_iswa_context specific API
    //

    const gptoss_kv_cache_context * get_base() const;
    const gptoss_kv_cache_context * get_swa()  const;

private:
    //gptoss_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<gptoss_ubatch> ubatches;

    const gptoss_memory_context_ptr ctx_base;
    const gptoss_memory_context_ptr ctx_swa;

    const gptoss_memory_status status;
};
