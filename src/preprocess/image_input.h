#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vinylizer {
inline cv::Mat normalize_input_image(const cv::Mat& input) {
    if (input.empty()) throw std::runtime_error("Cannot decode input image");
    cv::Mat image;
    if (input.depth() == CV_8U) image = input.clone();
    else if (input.depth() == CV_16U) input.convertTo(image, CV_8U, 1.0 / 257.0);
    else throw std::runtime_error("Only 8-bit and 16-bit unsigned images are supported");
    if (image.channels() == 1) cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    else if (image.channels() != 3 && image.channels() != 4)
        throw std::runtime_error("Unsupported image channel count");
    return image;
}

// Resize premultiplied color to avoid bleeding invisible RGB into alpha edges.
inline cv::Mat resize_input_image(const cv::Mat& image, int max_edge) {
    if (max_edge <= 0) throw std::invalid_argument("Resolution must be positive");
    const double scale = std::min(1.0, static_cast<double>(max_edge) / std::max(image.cols, image.rows));
    cv::Size size(std::max(1, static_cast<int>(image.cols * scale)),
                  std::max(1, static_cast<int>(image.rows * scale)));
    if (size == image.size()) return image.clone();
    cv::Mat resized;
    if (image.channels() != 4) {
        cv::resize(image, resized, size, 0, 0, cv::INTER_AREA);
        return resized;
    }
    cv::Mat floating;
    image.convertTo(floating, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(floating, channels);
    for (int c = 0; c < 3; ++c) channels[c] = channels[c].mul(channels[3]);
    for (auto& channel : channels) cv::resize(channel, channel, size, 0, 0, cv::INTER_AREA);
    cv::Mat divisor;
    cv::max(channels[3], 1e-8, divisor);
    for (int c = 0; c < 3; ++c) cv::divide(channels[c], divisor, channels[c]);
    cv::merge(channels, floating);
    floating.convertTo(resized, CV_8U, 255.0);
    return resized;
}
} // namespace vinylizer
