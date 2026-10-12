#pragma once

namespace vinylizer {

// Shared by the CUDA kernel and CPU numerical checks. Squared errors are
// unnormalized sums; gradients include the renderer's 4 * pixel_count divisor.
struct PixelLoss {
    float rgb_squared_error;
    float alpha_squared_error;
    float grad_rgb[3];
    float grad_alpha;
};

#if defined(__CUDACC__)
__host__ __device__
#endif
inline PixelLoss evaluate_pixel_loss(
    const float* output_rgb, const float* target_rgb,
    float opaque_mask, float rgb_weight,
    float alpha_difference, int pixel_count)
{
    PixelLoss loss{};
    const float scale = 2.0f / (pixel_count * 4.0f);

    // Transparent RGB has no meaningful target. Region importance affects
    // color fitting only, preserving the existing silhouette/alpha objective.
    if (opaque_mask >= 0.5f) {
        const float dr = output_rgb[0] - target_rgb[0];
        const float dg = output_rgb[1] - target_rgb[1];
        const float db = output_rgb[2] - target_rgb[2];
        loss.rgb_squared_error = (dr * dr + dg * dg + db * db) * rgb_weight;
        const float rgb_scale = scale * rgb_weight;
        loss.grad_rgb[0] = dr * rgb_scale;
        loss.grad_rgb[1] = dg * rgb_scale;
        loss.grad_rgb[2] = db * rgb_scale;
    }

    loss.alpha_squared_error = alpha_difference * alpha_difference;
    loss.grad_alpha = alpha_difference * scale;
    return loss;
}

} // namespace vinylizer
