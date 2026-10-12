#include "core/pixel_loss.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void expect_near(double actual, double expected, const std::string& label,
                 double absolute_tolerance = 1e-6,
                 double relative_tolerance = 1e-6) {
    const double tolerance = absolute_tolerance
        + relative_tolerance * std::max(std::abs(actual), std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(label + ": got " + std::to_string(actual)
                                 + ", expected " + std::to_string(expected));
    }
}

double objective(const std::array<float, 3>& output,
                 const std::array<float, 3>& target, float mask,
                 float weight, float alpha_difference, int pixel_count) {
    const auto loss = vinylizer::evaluate_pixel_loss(
        output.data(), target.data(), mask, weight, alpha_difference, pixel_count);
    return (static_cast<double>(loss.rgb_squared_error) + loss.alpha_squared_error)
        / (4.0 * pixel_count);
}

void unit_weight_preserves_baseline() {
    // Hand-calculated regression: RGB differences (4, -3, 2), alpha difference -5.
    const std::array<float, 3> output{14.0f, 17.0f, 32.0f};
    const std::array<float, 3> target{10.0f, 20.0f, 30.0f};
    const auto loss = vinylizer::evaluate_pixel_loss(
        output.data(), target.data(), 1.0f, 1.0f, -5.0f, 4);
    expect_near(loss.rgb_squared_error, 29.0, "baseline RGB SSE");
    expect_near(loss.alpha_squared_error, 25.0, "baseline alpha SSE");
    expect_near(objective(output, target, 1.0f, 1.0f, -5.0f, 4),
                3.375, "baseline RGBA mean");
    expect_near(loss.grad_rgb[0], 0.5, "baseline red gradient");
    expect_near(loss.grad_rgb[1], -0.375, "baseline green gradient");
    expect_near(loss.grad_rgb[2], 0.25, "baseline blue gradient");
    expect_near(loss.grad_alpha, -0.625, "baseline alpha gradient");

    // The existing opaque-mask boundary is inclusive at 0.5.
    const auto boundary = vinylizer::evaluate_pixel_loss(
        output.data(), target.data(), 0.5f, 1.0f, -5.0f, 4);
    expect_near(boundary.rgb_squared_error, 29.0, "opaque mask boundary");
}

void transparent_pixels_ignore_rgb() {
    // RGB values at transparent pixels may be arbitrary, even non-finite.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 3> output{nan, nan, nan};
    const std::array<float, 3> target{100.0f, -100.0f, 255.0f};
    for (const float mask : {0.0f, 0.49f}) {
        for (const float weight : {0.25f, 1.0f, 15.0f}) {
            const auto loss = vinylizer::evaluate_pixel_loss(
                output.data(), target.data(), mask, weight, 12.0f, 8);
            expect_near(loss.rgb_squared_error, 0.0, "transparent RGB SSE");
            for (const float gradient : loss.grad_rgb)
                expect_near(gradient, 0.0, "transparent RGB gradient");
            expect_near(loss.alpha_squared_error, 144.0, "transparent alpha SSE");
            expect_near(loss.grad_alpha, 0.75, "transparent alpha gradient");
        }
    }
}

void facial_priority_changes_only_rgb() {
    const std::array<float, 3> output{100.0f, 80.0f, 40.0f};
    const std::array<float, 3> target{80.0f, 110.0f, 30.0f};
    const auto background = vinylizer::evaluate_pixel_loss(
        output.data(), target.data(), 1.0f, 0.25f, -40.0f, 16);
    const auto feature = vinylizer::evaluate_pixel_loss(
        output.data(), target.data(), 1.0f, 3.0f, -40.0f, 16);
    expect_near(feature.rgb_squared_error / background.rgb_squared_error,
                12.0, "relative feature RGB SSE");
    for (int channel = 0; channel < 3; ++channel)
        expect_near(feature.grad_rgb[channel] / background.grad_rgb[channel],
                    12.0, "relative feature RGB gradient");
    expect_near(feature.alpha_squared_error, background.alpha_squared_error,
                "alpha SSE independent of facial priority");
    expect_near(feature.grad_alpha, background.grad_alpha,
                "alpha gradient independent of facial priority");
}

void gradients_match_finite_differences() {
    struct Case {
        std::array<float, 3> output;
        std::array<float, 3> target;
        float mask, weight, alpha_difference;
        int pixels;
    };
    const Case cases[] = {
        {{70.5f, 130.0f, 12.25f}, {35.0f, 170.5f, 14.0f}, 1.0f, 1.0f, -13.5f, 1},
        {{16.0f, 240.0f, 100.5f}, {20.0f, 200.0f, 125.0f}, 1.0f, 4.5f, 19.0f, 17},
        {{128.0f, 32.0f, 64.0f}, {100.0f, 50.0f, 40.0f}, 0.5f, 0.25f, -32.0f, 513},
        {{8.0f, 210.0f, 12.0f}, {208.0f, 25.0f, 122.0f}, 0.0f, 15.0f, 120.0f, 17},
        {{8.0f, 210.0f, 12.0f}, {208.0f, 25.0f, 122.0f}, 0.49f, 4.5f, -120.0f, 513},
        {{60.0f, 125.0f, 180.0f}, {60.0f, 125.0f, 180.0f}, 1.0f, 15.0f, 0.0f, 17},
    };
    constexpr float step = 0.125f;
    for (const auto& sample : cases) {
        const auto loss = vinylizer::evaluate_pixel_loss(
            sample.output.data(), sample.target.data(), sample.mask, sample.weight,
            sample.alpha_difference, sample.pixels);
        for (int channel = 0; channel < 4; ++channel) {
            auto plus = sample.output;
            auto minus = sample.output;
            float alpha_plus = sample.alpha_difference;
            float alpha_minus = sample.alpha_difference;
            if (channel < 3) {
                plus[channel] += step;
                minus[channel] -= step;
            } else {
                alpha_plus += step;
                alpha_minus -= step;
            }
            const double numerical = (
                objective(plus, sample.target, sample.mask, sample.weight,
                          alpha_plus, sample.pixels)
                - objective(minus, sample.target, sample.mask, sample.weight,
                            alpha_minus, sample.pixels)) / (2.0 * step);
            const float analytic = channel < 3 ? loss.grad_rgb[channel] : loss.grad_alpha;
            expect_near(analytic, numerical, "RGBA finite-difference gradient",
                        1e-4, 1e-3);
        }
    }
}

} // namespace

int main() {
    try {
        unit_weight_preserves_baseline();
        transparent_pixels_ignore_rgb();
        facial_priority_changes_only_rgb();
        gradients_match_finite_differences();
        std::cout << "pixel_loss_test: PASS (baseline, transparency, facial priority, "
                     "alpha invariance, 24 finite differences)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pixel_loss_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
