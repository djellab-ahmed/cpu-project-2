// SPDX-License-Identifier: MIT
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "gptoss-kv-view.h"

// Returns a sane default tile pad (power-of-two in [32, 2048]),
// honoring GPTOSS_KV_TILE if set.
int gptoss_kv_default_tile_pad(void);
