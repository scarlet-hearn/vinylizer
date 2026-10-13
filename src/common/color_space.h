#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define VIN_COLOR_HD __host__ __device__
#else
#define VIN_COLOR_HD
#endif

namespace vinylizer {
// Both encoded and linear channels use the same [0, 255] scale.
VIN_COLOR_HD inline float srgb_to_linear(float c) {
    const float v = c / 255.0f;
    return v <= 0.04045f ? c / 12.92f
        : 255.0f * powf((v + 0.055f) / 1.055f, 2.4f);
}

VIN_COLOR_HD inline float linear_to_srgb(float c) {
    const float v = fmaxf(c / 255.0f, 0.0f);
    return v <= 0.0031308f ? v * (12.92f * 255.0f)
        : 255.0f * (1.055f * powf(v, 1.0f / 2.4f) - 0.055f);
}

VIN_COLOR_HD inline float srgb_to_linear_derivative(float c) {
    const float v = c / 255.0f;
    return v <= 0.04045f ? 1.0f / 12.92f
        : (2.4f / 1.055f) * powf((v + 0.055f) / 1.055f, 1.4f);
}

// Derivative of the output transfer function, evaluated at its encoded result.
VIN_COLOR_HD inline float output_to_linear_gradient(float grad, float encoded_output) {
    return grad / srgb_to_linear_derivative(encoded_output);
}
} // namespace vinylizer

#undef VIN_COLOR_HD
