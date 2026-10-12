#pragma once

#include "common/types.h"
#include <string>
#include <functional>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cv { class Mat; }

namespace vinylizer {

struct PipelineConfig {
    std::string input_path;
    std::string output_path;
    int num_shapes = 500;
    int canvas_w = 512;
    int canvas_h = 512;
    int num_cycles = 3;
    float lr_start = 0.1f;
    float lr_end = 0.01f;
};

inline void validate_pipeline_config(const PipelineConfig& c) {
    if (c.num_shapes <= 0 || c.num_shapes > 65535 || c.canvas_w <= 0 || c.canvas_h <= 0
        || c.num_cycles <= 0 || c.num_cycles > std::numeric_limits<int>::max() / 200
        || !std::isfinite(c.lr_start) || !std::isfinite(c.lr_end)
        || c.lr_start <= 0 || c.lr_end <= 0 || c.lr_end > c.lr_start)
        throw std::invalid_argument("Invalid layers, resolution, cycles or learning rate");
}

struct PipelineCallbacks {
    const std::atomic<bool>* stop = nullptr;
    std::function<void(float, float, int)> progress;
    std::function<void(const cv::Mat&)> frame;
};

// Throws on load, optimization or output errors. GUI and CLI share this path.
void run_pipeline(const PipelineConfig& config, const PipelineCallbacks& callbacks = {},
                  bool write_preview = true);

} // namespace vinylizer
