#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace vinylizer::gui {

struct OptimizationInputs {
    int canvas_w;
    int canvas_h;
    int num_cycles;
    int num_shapes;
    float lr_start;
    float lr_end;
};

namespace detail {
inline std::optional<int> positive_integer(std::string_view text, int maximum) {
    if (text.empty()) return std::nullopt;
    for (const char c : text) {
        if (c < '0' || c > '9') return std::nullopt;
    }
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()
        || value <= 0 || value > maximum)
        return std::nullopt;
    return value;
}

inline std::optional<float> positive_float(std::string_view text) {
    if (text.empty()) return std::nullopt;
    float value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value,
                                        std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()
        || !std::isfinite(value) || value <= 0)
        return std::nullopt;
    return value;
}
} // namespace detail

// Parse the complete current text. An invalid edit has no usable value, so it
// cannot silently reuse a previously accepted layer count or configuration.
inline std::optional<int> parse_inject_layer_count(std::string_view text) {
    return detail::positive_integer(text, 65535);
}

inline std::optional<OptimizationInputs> parse_optimization_inputs(
    const std::array<std::string_view, 6>& text) {
    const int max_integer = (std::numeric_limits<int>::max)();
    const auto width = detail::positive_integer(text[0], max_integer);
    const auto height = detail::positive_integer(text[1], max_integer);
    // The pipeline expands each cycle into 200 optimization iterations.
    const auto cycles = detail::positive_integer(text[2], max_integer / 200);
    const auto layers = parse_inject_layer_count(text[3]);
    const auto start = detail::positive_float(text[4]);
    const auto end = detail::positive_float(text[5]);
    if (!width || !height || !cycles || !layers || !start || !end || *end > *start)
        return std::nullopt;
    return OptimizationInputs{*width, *height, *cycles, *layers, *start, *end};
}

inline std::optional<OptimizationInputs> parse_optimization_inputs(
    const std::array<std::string, 6>& text) {
    return parse_optimization_inputs(std::array<std::string_view, 6>{
        text[0], text[1], text[2], text[3], text[4], text[5]});
}

inline std::optional<OptimizationInputs> parse_optimization_inputs(
    const std::string (&text)[6]) {
    return parse_optimization_inputs(std::array<std::string_view, 6>{
        text[0], text[1], text[2], text[3], text[4], text[5]});
}

} // namespace vinylizer::gui
