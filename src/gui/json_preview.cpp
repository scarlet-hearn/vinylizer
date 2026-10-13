#include "gui/json_preview.h"
#include "common/canvas.h"
#include "common/path_utils.h"
#include "inject/inject.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vinylizer {
namespace {

constexpr int MAX_PREVIEW_EDGE = 2048;

bool read_dimension(const nlohmann::json& document, const char* key, double& value) {
    if (!document.contains(key) || !document[key].is_number()) return false;
    value = document[key].get<double>();
    return std::isfinite(value) && value >= 1 && std::floor(value) == value;
}

} // namespace

QImage render_json_preview_file(const std::string& json_path) {
    try {
        const auto shapes = validate_json(json_path);
        if (shapes.empty()) return {};

        auto file = open_ifstream(json_path);
        const auto document = nlohmann::json::parse(file);
        double width = 0, height = 0, origin_x = 0, origin_y = 0;
        if (document.contains("width") || document.contains("height")) {
            // Exported geometry is in canvas coordinates. Shapes extending
            // outside it must be clipped, without shifting or widening it.
            if (!read_dimension(document, "width", width)
                || !read_dimension(document, "height", height)) return {};
        } else {
            double min_x = std::numeric_limits<double>::infinity(), min_y = min_x;
            double max_x = -min_x, max_y = -min_x;
            for (const auto& shape : shapes) {
                const auto& data = shape["data"];
                double rx = data[2].get<double>(), ry = data[3].get<double>();
                if (shape["type"] == SHAPE_RECTANGLE) { rx *= 0.5; ry *= 0.5; }
                const double angle = data.size() >= 5 ? data[4].get<double>() : 0;
                const double radians = std::fmod(angle, 360.0) * 3.14159265358979323846 / 180.0;
                const double cos_a = std::cos(radians), sin_a = std::sin(radians);
                const double hx = std::abs(rx * cos_a) + std::abs(ry * sin_a);
                const double hy = std::abs(rx * sin_a) + std::abs(ry * cos_a);
                const double cx = data[0].get<double>(), cy = data[1].get<double>();
                min_x = std::min(min_x, cx - hx);
                min_y = std::min(min_y, cy - hy);
                max_x = std::max(max_x, cx + hx);
                max_y = std::max(max_y, cy + hy);
            }
            constexpr double padding = 4;
            origin_x = min_x - padding;
            origin_y = min_y - padding;
            width = max_x - min_x + padding * 2;
            height = max_y - min_y + padding * 2;
        }

        const double extent = std::max(width, height);
        if (!std::isfinite(extent) || extent <= 0) return {};
        const double scale = std::min(1.0, MAX_PREVIEW_EDGE / extent);
        // Clamp before the integer conversion, including rounding at the cap.
        const int preview_w = static_cast<int>(std::clamp(std::ceil(width * scale), 1.0, double(MAX_PREVIEW_EDGE)));
        const int preview_h = static_cast<int>(std::clamp(std::ceil(height * scale), 1.0, double(MAX_PREVIEW_EDGE)));

        std::vector<Shape> preview_shapes;
        preview_shapes.reserve(shapes.size());
        for (const auto& source : shapes) {
            const auto& data = source["data"];
            const auto& rgba = source["color"];
            Shape shape;
            shape.type = source["type"].get<int>();
            shape.cx = static_cast<float>((data[0].get<double>() - origin_x) * scale);
            shape.cy = static_cast<float>((data[1].get<double>() - origin_y) * scale);
            shape.rx = static_cast<float>(data[2].get<double>() * scale);
            shape.ry = static_cast<float>(data[3].get<double>() * scale);
            shape.angle = data.size() >= 5 ? static_cast<float>(std::fmod(data[4].get<double>(), 360.0)) : 0;
            shape.color = Color(rgba[0].get<uint8_t>(), rgba[1].get<uint8_t>(),
                                rgba[2].get<uint8_t>(), rgba[3].get<uint8_t>());
            shape.opacity = shape.color.a / 255.0f;
            preview_shapes.push_back(shape);
        }

        Canvas canvas(preview_w, preview_h);
        canvas.add_shapes(preview_shapes);
        // Canvas stores RGB channels, despite using an OpenCV matrix container.
        const cv::Mat rgb = canvas.clone_image();
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    } catch (...) {
        return {};
    }
}

} // namespace vinylizer
