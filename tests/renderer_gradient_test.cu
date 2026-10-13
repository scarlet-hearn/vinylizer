#include "core/diff_renderer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using vinylizer::ShapeParams;
using vinylizer::DiffRenderer;

void near(float actual, float expected, float absolute, float relative, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > absolute + relative * std::abs(expected))
        throw std::runtime_error(std::string(label) + ": " + std::to_string(actual)
                                 + " != " + std::to_string(expected));
}
float& parameter(ShapeParams& p, int index) {
    switch (index) {
        case 0: return p.cx; case 1: return p.cy;
        case 2: return p.rx; case 3: return p.ry;
        case 4: return p.angle; case 5: return p.color_r;
        case 6: return p.color_g; case 7: return p.color_b;
        default: return p.opacity;
    }
}
float evaluate(DiffRenderer& renderer, const std::vector<ShapeParams>& shapes) {
    renderer.upload_params_gpu(shapes.data(), static_cast<int>(shapes.size()));
    return renderer.render_forward_backward(static_cast<int>(shapes.size()));
}
void check_renderer(int tile_limit) {
    constexpr int W = 4, H = 3, N = 3, pixels = W * H;
    DiffRenderer renderer;
    renderer.init(W, H, N, tile_limit);
    renderer.set_bg_color(5, 40, 110);
    std::vector<float> target(pixels * 3), alpha(pixels, 200), mask(pixels, 1), weights(pixels, 1);
    for (int i = 0; i < pixels; ++i) {
        target[3 * i] = 130; target[3 * i + 1] = 65; target[3 * i + 2] = 95;
        weights[i] = i % 2 ? 1.8f : 0.2f;
    }
    renderer.set_target(target.data(), mask.data(), alpha.data(), weights.data());
    std::vector<ShapeParams> shapes = {
        {1.3f, 1.7f, 5.1f, 3.7f, 17, 50, 170, 90, 153.0f / 255},
        {0.2f, 0.6f, 4.3f, 6.1f, 63, 150, 6, 170, 204.0f / 255},
        {2.8f, 1.4f, 3.5f, 5.3f, 39, 230, 85, 25, 102.0f / 255}
    };
    const float baseline = evaluate(renderer, shapes);
    std::vector<float> gradients(N * 9);
    renderer.download_grads_packed(gradients.data(), N);
    for (int i = 0; i < N; ++i) {
        for (int p = 0; p < 9; ++p) {
            const float step = p == 8 ? 1.0f / 255 : p >= 5 ? 1.0f : 0.02f;
            auto plus = shapes, minus = shapes;
            parameter(plus[i], p) += step;
            parameter(minus[i], p) -= step;
            const float numerical = (evaluate(renderer, plus) - evaluate(renderer, minus)) / (2 * step);
            near(gradients[i * 9 + p], numerical, 0.12f, 0.05f, "CUDA finite difference");
        }
    }
    // Fully opaque pixels must not poison scratch or later evaluations.
    auto opaque = shapes;
    for (auto& shape : opaque) { shape.cx = 1; shape.cy = 1; shape.opacity = 1; }
    evaluate(renderer, opaque);
    renderer.download_grads_packed(gradients.data(), N);
    for (float gradient : gradients)
        if (!std::isfinite(gradient)) throw std::runtime_error("Opaque-layer gradient is not finite");
    near(evaluate(renderer, shapes), baseline, 0.01f, 1e-5f, "scratch reused after occlusion");
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "renderer_gradient_test: SKIP (no CUDA device)\n";
        return 77;
    }
    try {
        check_renderer(0);
        check_renderer(5); // Multiple tiles, partial rows and partially filled blocks.
        std::cout << "renderer_gradient_test: PASS (full and tiled gradients, opaque scratch reuse)\n";
    } catch (const std::exception& e) {
        std::cerr << "renderer_gradient_test: FAIL: " << e.what() << '\n';
        return 1;
    }
}
