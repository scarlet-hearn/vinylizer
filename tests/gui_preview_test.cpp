#include "gui/json_preview.h"
#include "common/path_utils.h"

#include <QColor>
#include <QTemporaryDir>
#include <cuda_runtime_api.h>
#include <json.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace vinylizer;
using Json = nlohmann::json;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Json rectangle(double x, double y, double w, double h, Json color) {
    return {{"type", 1}, {"data", {x, y, w, h}}, {"color", color}};
}

class PreviewFixture {
public:
    PreviewFixture() {
        require(directory_.isValid(), "Cannot create preview test directory");
        path_ = directory_.filePath(QString::fromUtf8("画布预览.json")).toUtf8().toStdString();
    }

    QImage render(const Json& document) { return render_text(document.dump()); }

    QImage render_text(const std::string& text) {
        require(write_file_bytes(path_, text.data(), text.size()), "Cannot write preview fixture");
        return render_json_preview_file(path_);
    }

private:
    QTemporaryDir directory_;
    std::string path_;
};

void require_color(const QImage& image, int x, int y, int red, int green, int blue, const char* message) {
    require(!image.isNull() && image.rect().contains(x, y), "Missing preview test pixel");
    const QColor color = image.pixelColor(x, y);
    require(std::abs(color.red() - red) <= 2 && std::abs(color.green() - green) <= 2
        && std::abs(color.blue() - blue) <= 2, message);
}

void canvas_tests(PreviewFixture& fixture) {
    Json document = {
        {"width", 40}, {"height", 24},
        {"shapes", Json::array({
            rectangle(20, 12, 40, 24, {0, 0, 255, 255}),
            rectangle(-2, 12, 16, 10, {255, 0, 0, 255}),
            rectangle(1000, 1000, 10, 10, {0, 255, 0, 255}),
            rectangle(20, 12, 4, 4, {255, 0, 0, 128}),
            rectangle(30, 12, 4, 4, {0, 255, 0, 255})
        })}
    };
    const auto image = fixture.render(document);
    require(image.width() == 40 && image.height() == 24, "Preview ignored the exported canvas dimensions");
    require_color(image, 0, 12, 255, 0, 0, "Partly outside shape was shifted or dropped");
    require_color(image, 8, 12, 0, 0, 255, "Shape clipping or RGB channel order changed");
    require_color(image, 39, 23, 0, 0, 255, "Canvas edge was cropped to shape bounds");
    require_color(image, 20, 12, 188, 0, 187, "Layer order or opacity was lost");
    require_color(image, 30, 12, 0, 255, 0, "Later shape was dropped from the preview");

    document["shapes"] = Json::array({rectangle(30, 12, 4, 4, {255, 0, 0, 255})});
    const auto sparse = fixture.render(document);
    require(sparse.size() == image.size(), "Sparse shapes changed the exported canvas dimensions");
    require_color(sparse, 0, 0, 0, 0, 0, "Canvas origin was shifted to the sparse shape");
    require_color(sparse, 30, 12, 255, 0, 0, "Sparse shape coordinates changed");
}

void legacy_and_size_tests(PreviewFixture& fixture) {
    Json document = {{"shapes", Json::array({rectangle(-10, 25, 20, 10, {255, 0, 0, 255})})}};
    const auto legacy = fixture.render(document);
    require(legacy.width() == 28 && legacy.height() == 18, "Legacy padded bounding box changed");
    require_color(legacy, 14, 9, 255, 0, 0, "Legacy coordinates were not translated into the preview");
    require_color(legacy, 0, 0, 0, 0, 0, "Legacy preview padding disappeared");

    document["width"] = 4096;
    document["height"] = 16;
    document["shapes"] = Json::array({rectangle(2048, 8, 4096, 16, {0, 255, 0, 255})});
    const auto capped = fixture.render(document);
    require(capped.width() == 2048 && capped.height() == 8, "Large canvas was not proportionally capped");
    require_color(capped, 1024, 4, 0, 255, 0, "Shape geometry did not scale with the canvas");

    document["width"] = 1e100;
    document["height"] = 1;
    const auto huge = fixture.render(document);
    require(huge.width() == 2048 && huge.height() == 1, "Huge dimensions overflowed preview allocation");

    document.erase("width");
    document.erase("height");
    document["shapes"] = Json::array({rectangle(0, 0, 1e30, 1e28, {255, 0, 0, 255})});
    const auto huge_legacy = fixture.render(document);
    require(huge_legacy.width() == 2048 && huge_legacy.height() == 21,
            "Legacy bounding box is not bounded before integer conversion");
}

void invalid_tests(PreviewFixture& fixture) {
    Json document = {{"width", 40}, {"height", 24},
        {"shapes", Json::array({rectangle(20, 12, 40, 24, {255, 0, 0, 255})})}};
    for (const auto& value : Json::array({0, -1, 1.5, "40", nullptr, true})) {
        auto invalid = document;
        invalid["width"] = value;
        require(fixture.render(invalid).isNull(), "Invalid canvas width was accepted");
        invalid = document;
        invalid["height"] = value;
        require(fixture.render(invalid).isNull(), "Invalid canvas height was accepted");
    }
    auto invalid = document;
    invalid.erase("height");
    require(fixture.render(invalid).isNull(), "Incomplete canvas dimensions fell back to shape bounds");
    invalid = document;
    invalid["shapes"][0]["data"] = {1, 2};
    require(fixture.render(invalid).isNull(), "Malformed shape bypassed validation");
    invalid = document;
    invalid["shapes"] = Json::array();
    require(fixture.render(invalid).isNull(), "Empty shape list was accepted");
    require(fixture.render(Json::array()).isNull(), "Invalid document root was accepted");
    require(fixture.render_text("{broken").isNull(), "Malformed JSON escaped preview validation");
    require(fixture.render_text("{\"width\":1e1000}").isNull(), "JSON numeric overflow escaped preview validation");
    require(render_json_preview_file("").isNull(), "Missing input was accepted");
}

} // namespace

int main() {
    try {
        PreviewFixture fixture;
        invalid_tests(fixture);
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "SKIP: CUDA device unavailable for GUI preview rendering\n";
            return 77;
        }
        canvas_tests(fixture);
        legacy_and_size_tests(fixture);
        std::cout << "GUI preview canvas, clipping, RGB, legacy and validation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
