#include "core/pipeline.h"
#include "common/logger.h"
#include "common/canvas.h"
#include "common/path_utils.h"
#include "preprocess/preprocessor.h"
#include "preprocess/image_input.h"
#include "core/optimizer.h"
#include "output/json_writer.h"
#include <opencv2/imgcodecs.hpp>
#include <chrono>

namespace vinylizer {
void run_pipeline(const PipelineConfig& config, const PipelineCallbacks& callbacks, bool write_preview) {
    validate_pipeline_config(config);
    auto bytes = read_file_bytes(config.input_path);
    if (bytes.empty()) throw std::runtime_error("Cannot read input image: " + config.input_path);
    cv::Mat input = normalize_input_image(cv::imdecode(bytes, cv::IMREAD_UNCHANGED));
    const int W = input.cols, H = input.rows;
    cv::Mat scaled_input = resize_input_image(input, std::max(config.canvas_w, config.canvas_h));
    Preprocessor preprocessor;
    auto prep = preprocessor.process(scaled_input);
    auto [r, g, b] = compute_bg_color_from_result(prep.posterized, prep.opaque_mask, prep.palette, prep.rgb_image);
    OptimizerConfig opt;
    opt.max_shapes = config.num_shapes;
    opt.opt_resolution = std::max(scaled_input.cols, scaled_input.rows);
    opt.total_iters = config.num_cycles * 200;
    opt.lr_start = config.lr_start;
    opt.lr_end = config.lr_end;
    GradientOptimizer optimizer(opt);
    optimizer.set_stop_flag(callbacks.stop);
    auto last_frame = std::chrono::steady_clock::now();
    optimizer.set_progress_callback([&](float progress, float mse, int iter) {
        if (callbacks.progress) callbacks.progress(progress, mse, iter);
        auto now = std::chrono::steady_clock::now();
        if (callbacks.frame && now - last_frame >= std::chrono::milliseconds(100)) {
            cv::Mat frame = optimizer.get_canvas();
            if (!frame.empty()) callbacks.frame(frame);
            last_frame = now;
        }
    });
    auto shapes = optimizer.optimize(prep.rgb_image, prep.opaque_mask, prep.edge_map,
        prep.saliency_map, prep.dark_mask, prep.posterized, prep.palette, prep.grad_dir,
        Color(r, g, b, 255), prep.alpha_mask);
    if (shapes.size() != static_cast<size_t>(config.num_shapes))
        throw std::runtime_error("Optimizer returned an unexpected number of shapes");
    const float sx = static_cast<float>(W) / scaled_input.cols;
    const float sy = static_cast<float>(H) / scaled_input.rows;
    for (auto& shape : shapes) {
        shape.cx *= sx; shape.cy *= sy; shape.rx *= sx; shape.ry *= sy;
    }
    write_json(config.output_path, shapes, W, H);
    if (write_preview) {
        Canvas canvas(W, H);
        canvas.add_shapes(shapes);
        // Canvas uses RGB; OpenCV image codecs expect BGR.
        cv::Mat bgr;
        cv::cvtColor(canvas.clone_image(), bgr, cv::COLOR_RGB2BGR);
        std::vector<uint8_t> encoded;
        if (!cv::imencode(".png", bgr, encoded)
            || !write_file_bytes(config.output_path + ".preview.png", encoded.data(), encoded.size()))
            throw std::runtime_error("Cannot save preview image");
    }
}
} // namespace vinylizer
