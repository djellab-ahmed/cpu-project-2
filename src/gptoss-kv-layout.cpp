// SPDX-License-Identifier: MIT
#include "gptoss-kv-layout.h"

#include <cstdlib>

int gptoss_kv_default_tile_pad(void) {
    const char *v = std::getenv("GPTOSS_KV_TILE");
    int pad = v ? std::atoi(v) : 128;
    if (pad < 32) {
        pad = 32;
    }
    if (pad > 2048) {
        pad = 2048;
    }
    // round to power of two for nicer modulo arithmetic
    int p = 1;
    while (p < pad) {
        p <<= 1;
    }
    return p;
}

