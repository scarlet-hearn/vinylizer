#pragma once
#include <cmath>

namespace vinylizer {
// Match the CUDA renderer's linear-light compositing, with sRGB bytes at the API.
inline float composite_target_color(float color, float background, float alpha) {
    if (alpha <= 0.0f) return background;
    if (alpha >= 1.0f) return color;
    auto linear = [](float v) {
        v /= 255.0f;
        return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    };
    const float v = alpha * linear(color) + (1.0f - alpha) * linear(background);
    return 255.0f * (v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f);
}
} // namespace vinylizer
