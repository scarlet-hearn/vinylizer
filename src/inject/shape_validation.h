#pragma once
#include <json.hpp>
#include <cmath>
#include <limits>
#include <string>

namespace vinylizer {
constexpr int MAX_LAYER_COUNT = 65535; // The game's layer count is a uint16.

inline bool valid_layer_count(int count, size_t shape_count) {
    return count > 0 && count <= MAX_LAYER_COUNT && static_cast<size_t>(count) == shape_count;
}

inline bool validate_shape(const nlohmann::json& shape, std::string& error) {
    error.clear();
    if (!shape.is_object() || !shape.contains("type") || !shape["type"].is_number_integer()
        || (shape["type"] != 1 && shape["type"] != 16 && shape["type"] != 228)) {
        error = "Shape type must be 1, 16 or 228";
        return false;
    }
    const bool rect = shape["type"] == 1;
    if (!shape.contains("data") || !shape["data"].is_array()
        || (shape["data"].size() != 5 && !(rect && shape["data"].size() == 4))) {
        error = "Shape data must have 4 values for a rectangle or 5 for an ellipse";
        return false;
    }
    for (const auto& value : shape["data"]) {
        if (!value.is_number()) { error = "Geometry must be numeric"; return false; }
        const double number = value.get<double>();
        if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max()) {
            error = "Geometry must be finite and representable as float";
            return false;
        }
    }
    if (shape["data"][2].get<double>() < 0 || shape["data"][3].get<double>() < 0) {
        error = "Shape dimensions cannot be negative";
        return false;
    }
    if (!shape.contains("color") || !shape["color"].is_array() || shape["color"].size() != 4) {
        error = "Color must have exactly 4 RGBA values";
        return false;
    }
    for (const auto& value : shape["color"]) {
        if (!value.is_number_integer() || value.get<double>() < 0 || value.get<double>() > 255) {
            error = "RGBA values must be integers in [0, 255]";
            return false;
        }
    }
    return true;
}
} // namespace vinylizer
