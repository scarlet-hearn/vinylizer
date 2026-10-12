#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace vinylizer {

struct SaliencyWeights {
    cv::Mat priority;  // CV_32F, [1, 15], neutral outside the opaque mask
    cv::Mat rgb_loss;  // CV_32F, mean 1 over opaque pixels
};

// Keep the total RGB loss scale stable as the number/size of faces changes.
// Alpha loss is deliberately independent of facial importance.
inline SaliencyWeights build_saliency_weights(const cv::Mat& saliency,
                                              const cv::Mat& opaque_mask) {
    CV_Assert(!opaque_mask.empty() && opaque_mask.type() == CV_8UC1);
    SaliencyWeights weights;
    if (saliency.empty()) {
        weights.priority = cv::Mat::ones(opaque_mask.size(), CV_32FC1);
    } else {
        CV_Assert(saliency.channels() == 1);
        cv::Mat input;
        saliency.convertTo(input, CV_32F);
        // Sanitize before interpolation so a bad pixel cannot taint its neighbors.
        for (int y = 0; y < input.rows; ++y) {
            float* row = input.ptr<float>(y);
            for (int x = 0; x < input.cols; ++x)
                row[x] = std::isfinite(row[x]) ? std::clamp(row[x], 1.0f, 15.0f) : 1.0f;
        }
        cv::resize(input, weights.priority, opaque_mask.size(), 0, 0, cv::INTER_LINEAR);
        weights.priority.setTo(1.0f, opaque_mask == 0);
    }

    weights.rgb_loss = cv::Mat::ones(opaque_mask.size(), CV_32FC1);
    if (cv::countNonZero(opaque_mask) > 0) {
        const float mean_weight = static_cast<float>(cv::mean(weights.priority, opaque_mask)[0]);
        weights.rgb_loss = weights.priority / mean_weight;
        weights.rgb_loss.setTo(1.0f, opaque_mask == 0);
    }
    return weights;
}

// Smaller initial primitives give high-priority features useful detail immediately.
inline float saliency_radius_scale(float priority) {
    return 1.0f / std::sqrt(std::clamp(priority, 1.0f, 12.0f));
}

} // namespace vinylizer
