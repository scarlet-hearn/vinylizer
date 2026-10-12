#include "inject/inject.h"
#include "output/json_writer.h"
#include "core/pipeline.h"
#include "core/alpha_target.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <filesystem>

using namespace vinylizer;
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
template<class Fn> void rejects(Fn fn, const char* text) {
    bool failed = false;
    try { fn(); } catch (const std::exception&) { failed = true; }
    require(failed, text);
}
nlohmann::json shape() { return {{"type", 228}, {"data", {10, 20, 5, 6, 30}}, {"color", {20, 40, 60, 128}}}; }

void schema_tests() {
    std::string error;
    auto good = shape();
    require(validate_shape(good, error), "Valid shape rejected");
    for (const auto& bad : std::vector<nlohmann::json>{nullptr, 7, "shape", nlohmann::json::array()})
        require(!validate_shape(bad, error), "Non-object accepted");
    for (const auto& key : {"type", "data", "color"}) {
        auto bad = good; bad.erase(key);
        require(!validate_shape(bad, error), "Missing required field accepted");
    }
    auto bad = good; bad["type"] = "228";
    require(!validate_shape(bad, error), "String type accepted");
    bad = good; bad["data"] = {1, 2};
    require(!validate_shape(bad, error), "Short data accepted");
    for (auto number : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), 1e100}) {
        bad = good; bad["data"][0] = number;
        require(!validate_shape(bad, error), "Nonfinite/overflow geometry accepted");
    }
    bad = good; bad["data"][2] = -1;
    require(!validate_shape(bad, error), "Negative dimensions accepted");
    for (double value : {-1.0, 256.0, 1.5}) {
        bad = good; bad["color"][0] = value;
        require(!validate_shape(bad, error), "Invalid color accepted");
    }
    require(valid_layer_count(80, 80) && !valid_layer_count(79, 80)
        && !valid_layer_count(81, 80) && !valid_layer_count(0, 0)
        && !valid_layer_count(65536, 65536), "Layer counts are not exact/bounded");
}

void pipeline_tests() {
    PipelineConfig config;
    validate_pipeline_config(config);
    config.num_shapes = 0;
    rejects([&] { validate_pipeline_config(config); }, "Zero layers accepted");
    config = {}; config.canvas_w = -1;
    rejects([&] { validate_pipeline_config(config); }, "Negative resolution accepted");
    config = {}; config.lr_end = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { validate_pipeline_config(config); }, "NaN learning rate accepted");
    config = {}; config.num_cycles = std::numeric_limits<int>::max();
    rejects([&] { validate_pipeline_config(config); }, "Iteration overflow accepted");
    require(composite_target_color(200, 40, 0) == 40 && composite_target_color(200, 40, 1) == 200,
            "Alpha endpoints changed");
    const float half = composite_target_color(255, 0, 0.5f);
    require(std::abs(half - 187.516f) < 0.01f, "Semi-transparent target is not linear-light composited");
}

void output_tests() {
    auto directory = std::filesystem::path("safety-test-output");
    std::filesystem::create_directories(directory);
    auto file = (directory / "valid.json").string();
    write_json(file, {Shape::make_soft_ellipse_228(10, 20, 5, 6, 30, Color(20, 40, 60), 128.0f / 255)}, 64, 64);
    require(validate_json(file).size() == 1, "JSON round trip failed");
    rejects([&] { write_json(directory.string(), {}, 64, 64); }, "Output-open failure did not propagate");
    { std::ofstream out(directory / "invalid.json"); out << R"({"shapes":[{"type":228,"data":[1,2],"color":[1,2,3,4]}]})"; }
    require(validate_json((directory / "invalid.json").string()).empty(), "Malformed file accepted");
    { std::ofstream out(directory / "overflow.json"); out << R"({"shapes":[{"type":228,"data":[1e1000]}]})"; }
    require(validate_json((directory / "overflow.json").string()).empty(), "Parser overflow escaped validation");
    // Remove only files explicitly created by this test.
    std::filesystem::remove(directory / "valid.json");
    std::filesystem::remove(directory / "invalid.json");
    std::filesystem::remove(directory / "overflow.json");
    std::filesystem::remove(directory);
}

void memory_tests() {
    // Only this test process's synthetic memory is accessed; no game process.
    constexpr int count = 80;
    auto* memory = static_cast<unsigned char*>(VirtualAlloc(nullptr, 32768, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    require(memory != nullptr, "Allocation failed");
    try {
        auto* table = reinterpret_cast<uint64_t*>(memory);
        for (int i = 0; i < count; ++i) {
            auto* layer = memory + 1024 + i * 128;
            table[i] = reinterpret_cast<uint64_t>(layer);
            layer[0x7A] = 102;
            const float scale[] = {1, 1};
            std::memcpy(layer + 0x28, scale, sizeof(scale));
        }
        const auto address = reinterpret_cast<uint64_t>(table);
        auto* group = memory + 30000;
        const uint16_t group_count = count;
        std::memcpy(group + 0x5A, &group_count, 2);
        std::memcpy(group + 0x78, &address, 8);
        require(layer_table_matches_h(GetCurrentProcess(), reinterpret_cast<uint64_t>(group), address, count),
                "Group metadata rejected");
        require(!layer_table_matches_h(GetCurrentProcess(), reinterpret_cast<uint64_t>(group), address, count - 1),
                "Changed count was not rejected");
        require(!layer_table_matches_h(GetCurrentProcess(), reinterpret_cast<uint64_t>(group), address + 8, count),
                "Changed table was not rejected");
        require(validate_layer_table_h(GetCurrentProcess(), address, count), "Valid synthetic table rejected");
        memory[1024 + 79 * 128 + 0x7A] = 228;
        require(!validate_layer_table_h(GetCurrentProcess(), address, count), "Layer after index 64 was not checked");
        memory[1024 + 79 * 128 + 0x7A] = 102;
        const auto saved = table[79]; table[79] = table[0];
        require(!validate_layer_table_h(GetCurrentProcess(), address, count), "Duplicate layer pointers accepted");
        table[79] = saved;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(memory + 1024 + 0x18, &nan, 4);
        require(!validate_layer_table_h(GetCurrentProcess(), address, count), "Nonfinite placeholder accepted");
        const float zero = 0; std::memcpy(memory + 1024 + 0x18, &zero, 4);
        require(write_layer_h(GetCurrentProcess(), address, 0, shape()), "Valid synthetic write failed");
        float y; std::memcpy(&y, memory + 1024 + 0x1C, 4);
        require(y == -20 && memory[1024 + 0x7A] == 228, "Shape conversion was not written correctly");
        auto before = std::array<unsigned char, 128>{};
        std::memcpy(before.data(), memory + 1024, 128);
        require(!write_layer_h(GetCurrentProcess(), address, 0, shape()), "Non-placeholder overwrite accepted");
        require(std::memcmp(before.data(), memory + 1024, 128) == 0, "Rejected write changed memory");
        DWORD old;
        require(VirtualProtect(memory, 32768, PAGE_READONLY, &old), "Protection change failed");
        require(!validate_layer_table_h(GetCurrentProcess(), address, count), "Read-only table accepted");
    } catch (...) { VirtualFree(memory, 0, MEM_RELEASE); throw; }
    VirtualFree(memory, 0, MEM_RELEASE);
}

int main() {
    try {
        schema_tests(); pipeline_tests(); output_tests(); memory_tests();
        std::cout << "PASS: schema, config, alpha, output errors, full-table checks and synthetic memory writes\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
