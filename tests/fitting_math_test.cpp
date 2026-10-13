#include "core/compositing_gradient.h"
#include "core/residual_placement.h"

#include <array>
#include <iostream>
#include <random>
#include <set>
#include <string>

namespace {
using Layer = std::array<double, 9>; // cx, cy, rx, ry, angle, RGB, opacity
using RGB = std::array<double, 3>;
constexpr double pi = 3.14159265358979323846;

void require(bool value, const char* label) {
    if (!value) throw std::runtime_error(label);
}
void near(double actual, double expected, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 0.01 + 0.002 * std::abs(expected))
        throw std::runtime_error(std::string(label) + ": " + std::to_string(actual)
                                 + " != " + std::to_string(expected));
}

// Independent double-precision forward reference, with no production derivative
// helpers or quantization. STE differentiates this continuous rendering model.
double linear(double c) {
    return c <= 255.0 * 0.04045 ? c / 12.92
        : 255.0 * std::pow((c / 255.0 + 0.055) / 1.055, 2.4);
}
double encoded(double c) {
    return c <= 255.0 * 0.0031308 ? c * 12.92
        : 255.0 * (1.055 * std::pow(c / 255.0, 1.0 / 2.4) - 0.055);
}
double profile(const Layer& p, double x, double y) {
    const double a = p[4] * pi / 180, dx = x - p[0], dy = y - p[1];
    const double u = (dx * std::cos(a) + dy * std::sin(a)) / std::max(0.5, p[2]);
    const double v = (-dx * std::sin(a) + dy * std::cos(a)) / std::max(0.5, p[3]);
    const double r = std::hypot(u, v);
    if (r >= 1) return 0;
    double alpha = 1 - r;
    for (int k = 0; k < 8; ++k)
        alpha += (1 - 0.9132 * alpha - 0.1875 * alpha * alpha
                  + 0.1007 * alpha * alpha * alpha - r)
            / (0.9132 + 0.375 * alpha - 0.3021 * alpha * alpha);
    return std::clamp(alpha, 0.0, 1.0);
}
std::array<double, 4> render(const std::vector<Layer>& layers, double x, double y, RGB bg) {
    for (double& c : bg) c = linear(c);
    double transmittance = 1;
    for (const auto& p : layers) {
        const double alpha = profile(p, x, y) * p[8];
        for (int c = 0; c < 3; ++c) bg[c] = alpha * linear(p[5 + c]) + (1 - alpha) * bg[c];
        transmittance *= 1 - alpha;
    }
    return {encoded(bg[0]), encoded(bg[1]), encoded(bg[2]), 255 * (1 - transmittance)};
}
double objective(const std::vector<Layer>& layers, double x, double y, RGB bg,
                 const std::array<double, 4>& target, double rgb_weight) {
    const auto output = render(layers, x, y, bg);
    double sum = 0;
    for (int c = 0; c < 4; ++c)
        sum += (c < 3 ? rgb_weight : 1) * std::pow(output[c] - target[c], 2) / 4;
    return sum;
}
std::vector<Layer> gradients(const std::vector<Layer>& layers, double x, double y, RGB bg,
                            const std::array<double, 4>& target, double rgb_weight) {
    const auto output = render(layers, x, y, bg);
    float grad_linear[3], below[3];
    for (int c = 0; c < 3; ++c) {
        grad_linear[c] = vinylizer::output_to_linear_gradient(
            static_cast<float>((output[c] - target[c]) * rgb_weight / 2),
            static_cast<float>(output[c]));
        below[c] = static_cast<float>(linear(bg[c]));
    }
    std::vector<float> above(layers.size());
    float transmittance = 1;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        above[i] = transmittance;
        transmittance *= 1 - static_cast<float>(profile(layers[i], x, y) * layers[i][8]);
    }
    float below_t = 1;
    std::vector<Layer> result(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& p = layers[i];
        const float ar = static_cast<float>(profile(p, x, y)), ae = ar * static_cast<float>(p[8]);
        const float color[] = {static_cast<float>(p[5]), static_cast<float>(p[6]), static_cast<float>(p[7])};
        const auto blend = vinylizer::composite_layer_gradient(grad_linear,
            static_cast<float>((output[3] - target[3]) / 2), color, below, ae, above[i], below_t);
        const auto geometry = vinylizer::ellipse_alpha_gradient(
            static_cast<float>(x - p[0]), static_cast<float>(y - p[1]),
            static_cast<float>(p[2]), static_cast<float>(p[3]),
            static_cast<float>(std::sin(p[4] * pi / 180)),
            static_cast<float>(std::cos(p[4] * pi / 180)), ar);
        const double factor = blend.effective_alpha * p[8];
        result[i] = {factor * geometry.cx, factor * geometry.cy, factor * geometry.rx,
            factor * geometry.ry, factor * geometry.angle, blend.color[0], blend.color[1],
            blend.color[2], ar * blend.effective_alpha};
        for (int c = 0; c < 3; ++c) below[c] = ae * vinylizer::srgb_to_linear(color[c]) + (1 - ae) * below[c];
        below_t *= 1 - ae;
    }
    return result;
}

int finite_differences() {
    int checks = 0;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> unit(0, 1);
    for (int sample = 0; sample < 24; ++sample) {
        std::vector<Layer> layers;
        for (int i = 0; i < 3; ++i)
            layers.push_back({8 + unit(rng) * 3, 8 + unit(rng) * 3,
                5 + unit(rng) * 4, 4 + unit(rng) * 3, unit(rng) * 360,
                1 + unit(rng) * 250, 1 + unit(rng) * 250, 1 + unit(rng) * 250,
                0.2 + unit(rng) * 0.7});
        if (sample == 0) layers[1] = {10, 10, 8, 6, 30, 100, 2, 230, 1};
        if (sample == 1) layers[2][8] = 0; // zero opacity must still receive a gradient
        if (sample == 2) layers[0] = {50, 50, 2, 3, 0, 4, 8, 2, 0.5}; // outside support
        const RGB bg{3, 40, 120};
        const std::array<double, 4> target{145, 7, 95, 165};
        const double weight = sample % 3 == 0 ? 0 : sample % 3 == 1 ? 1 : 12;
        const auto analytic = gradients(layers, 10, 10, bg, target, weight);
        for (size_t i = 0; i < layers.size(); ++i) {
            for (int p = 0; p < 9; ++p) {
                const double h = p == 8 ? 1e-5 : 1e-3;
                auto plus = layers, minus = layers;
                plus[i][p] += h; minus[i][p] -= h;
                const double numerical = (objective(plus, 10, 10, bg, target, weight)
                    - objective(minus, 10, 10, bg, target, weight)) / (2 * h);
                near(analytic[i][p], numerical, "layer finite difference");
                ++checks;
            }
        }
    }
    return checks;
}

void placement_checks() {
    constexpr int W = 41, H = 41;
    std::vector<float> residual(W * H, 0);
    require(vinylizer::select_residual_seeds(residual.data(), W, H, 100, 1, 8).empty(), "zero residual");
    for (int dx = -6; dx <= 6; ++dx) residual[10 * W + 10 + dx] = 100 - 2.0f * std::abs(dx);
    residual[30 * W + 30] = 90;
    auto seeds = vinylizer::select_residual_seeds(residual.data(), W, H, 2, 1, 8);
    require(seeds.size() == 2, "fixed replacement budget");
    require(seeds[0].x == 10 && seeds[0].y == 10, "residual maximum");
    require(seeds[0].rx > seeds[0].ry * 3 && std::abs(seeds[0].angle) < 1e-4f, "horizontal detail orientation");
    require(seeds[1].x == 30 && seeds[1].y == 30, "spread replacements across distinct errors");
    std::fill(residual.begin(), residual.end(), 0.0f);
    for (int d = -6; d <= 6; ++d) residual[(20 + d) * W + 20 + d] = 100 - 2.0f * std::abs(d);
    seeds = vinylizer::select_residual_seeds(residual.data(), W, H, 1, 1, 8);
    near(seeds[0].angle, 45, "diagonal detail orientation");
    const float original_radius = seeds[0].rx;
    std::vector<float> detail_scale(W * H, 0.5f);
    seeds = vinylizer::select_residual_seeds(residual.data(), W, H, 1, 1, 8, detail_scale.data());
    near(seeds[0].rx, original_radius / 2, "facial detail radius");
    float tiny[] = {3, 2, 1, std::numeric_limits<float>::quiet_NaN()};
    seeds = vinylizer::select_residual_seeds(tiny, 2, 2, 1000, 1, 4);
    std::set<std::pair<int, int>> positions;
    for (const auto& seed : seeds) positions.emplace(seed.x, seed.y);
    require(positions.size() == 3 && seeds.size() == 3, "unique finite seeds in tiny images");
    const float single = 1;
    seeds = vinylizer::select_residual_seeds(&single, 1, 1, 50, 1, 2);
    require(seeds.size() == 1 && seeds[0].x == 0 && seeds[0].y == 0, "single pixel");
}
} // namespace

int main() {
    try {
        const int checks = finite_differences();
        placement_checks();
        std::cout << "fitting_math_test: PASS (" << checks
                  << " multilayer finite differences; occlusion, linear color, alpha, geometry, residual coverage)\n";
    } catch (const std::exception& e) {
        std::cerr << "fitting_math_test: FAIL: " << e.what() << '\n';
        return 1;
    }
}
