#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct gptoss_file;
struct gptoss_mmap;
struct gptoss_mlock;

using gptoss_files  = std::vector<std::unique_ptr<gptoss_file>>;
using gptoss_mmaps  = std::vector<std::unique_ptr<gptoss_mmap>>;
using gptoss_mlocks = std::vector<std::unique_ptr<gptoss_mlock>>;

struct gptoss_file {
    gptoss_file(const char * fname, const char * mode);
    ~gptoss_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len) const;
    uint32_t read_u32() const;

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct gptoss_mmap {
    gptoss_mmap(const gptoss_mmap &) = delete;
    gptoss_mmap(struct gptoss_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~gptoss_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct gptoss_mlock {
    gptoss_mlock();
    ~gptoss_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t gptoss_path_max();
