#pragma once

#include <cmath>

static inline float max_abs(const float * x, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        float a = std::fabs(x[i]);
        if (a > m) {
            m = a;
        }
    }
    return m;
}

static inline float safe_scale(float amax) {
    float s = amax / 127.0f;
    return s < 1e-8f ? 1e-8f : s;
}

