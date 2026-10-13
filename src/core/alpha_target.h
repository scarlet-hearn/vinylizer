#pragma once
#include "common/color_space.h"

namespace vinylizer {
// Match the CUDA renderer's linear-light compositing, with sRGB bytes at the API.
inline float composite_target_color(float color, float background, float alpha) {
    if (alpha <= 0.0f) return background;
    if (alpha >= 1.0f) return color;
    return linear_to_srgb(alpha * srgb_to_linear(color)
        + (1.0f - alpha) * srgb_to_linear(background));
}
} // namespace vinylizer
