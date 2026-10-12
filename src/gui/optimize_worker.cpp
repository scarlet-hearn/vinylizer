#include "gui/optimize_worker.h"
#include "common/logger.h"
#include <opencv2/core.hpp>
#include <exception>

namespace vinylizer {
OptimizeWorker::OptimizeWorker(const PipelineConfig& config, QObject* parent)
    : QThread(parent), config_(config) {}

void OptimizeWorker::run() {
    try {
        PipelineCallbacks callbacks;
        callbacks.stop = &stop_requested_;
        callbacks.progress = [this](float pct, float mse, int iter) { emit progress(pct, mse, iter); };
        callbacks.frame = [this](const cv::Mat& rgb) {
            QImage image(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
            emit frame_ready(image.copy());
        };
        // Other preview threads may still own CUDA buffers: never reset the device here.
        run_pipeline(config_, callbacks, false);
        emit done(true, stop_requested_.load(), QString());
    } catch (const std::exception& error) {
        VIN_ERROR("Optimization failed: %s", error.what());
        emit done(false, false, QString::fromUtf8(error.what()));
    } catch (...) {
        emit done(false, false, QString::fromUtf8("未知优化错误"));
    }
}
} // namespace vinylizer
