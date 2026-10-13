#pragma once
#include "common/color_space.h"
#include "common/alpha_228.h"

#if defined(__CUDACC__)
#define VIN_GRAD_HD __host__ __device__
#else
#define VIN_GRAD_HD
#endif

namespace vinylizer {
struct LayerGradient {
    float color[3];
    float effective_alpha;
};

// Above/below transmittance are products excluding this layer. Explicit products
// handle fully opaque layers without log(0) or division by (1 - alpha).
VIN_GRAD_HD inline LayerGradient composite_layer_gradient(
    const float* grad_linear, float grad_alpha, const float* encoded_color,
    const float* below_linear, float effective_alpha,
    float above_transmittance, float below_transmittance)
{
    LayerGradient result{};
    result.effective_alpha = grad_alpha * 255.0f * above_transmittance * below_transmittance;
    for (int c = 0; c < 3; ++c) {
        result.color[c] = grad_linear[c] * effective_alpha * above_transmittance
            * srgb_to_linear_derivative(encoded_color[c]);
        result.effective_alpha += grad_linear[c] * above_transmittance
            * (srgb_to_linear(encoded_color[c]) - below_linear[c]);
    }
    return result;
}

struct EllipseGradient { float cx, cy, rx, ry, angle; };

// Derivative of the smooth 228 profile. Rotation is parameterized in degrees.
VIN_GRAD_HD inline EllipseGradient ellipse_alpha_gradient(
    float dx, float dy, float rx, float ry, float sine, float cosine,
    float profile_alpha)
{
    const float u = dx * cosine + dy * sine, v = -dx * sine + dy * cosine;
    const float rxc = fmaxf(rx, 0.5f), ryc = fmaxf(ry, 0.5f);
    const float rx2 = rxc * rxc, ry2 = ryc * ryc;
    const float rn = sqrtf(u * u / rx2 + v * v / ry2);
    if (rn >= 1.0f || profile_alpha <= 0.0f) return {};
    const float a = profile_alpha;
    const float denom = C1_228 + 2.0f * C2_228 * a + 3.0f * C3_228 * a * a;
    const float scale = -1.0f / (fmaxf(denom, 1e-10f) * 2.0f * fmaxf(rn, 1e-6f));
    return {
        scale * -2.0f * (u * cosine / rx2 - v * sine / ry2),
        scale * -2.0f * (u * sine / rx2 + v * cosine / ry2),
        rx > 0.5f ? scale * -2.0f * u * u / (rxc * rx2) : 0.0f,
        ry > 0.5f ? scale * -2.0f * v * v / (ryc * ry2) : 0.0f,
        scale * 2.0f * u * v * (3.14159265358979323846f / 180.0f) * (1.0f / rx2 - 1.0f / ry2)
    };
}
} // namespace vinylizer

#undef VIN_GRAD_HD
