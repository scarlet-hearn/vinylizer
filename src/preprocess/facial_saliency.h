#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace vinylizer {

struct FacialSaliencyConfig {
    float face_weight = 3.0f;
    float eye_weight = 12.0f;
    float mouth_weight = 10.0f;
    float fallback_eye_weight = 7.0f;
    float fallback_mouth_weight = 6.0f;
};

// All rectangles use image coordinates. Eyes/mouths belong to this face.
struct FacialRegions {
    cv::Rect face;
    std::vector<cv::Rect> eyes;
    std::vector<cv::Rect> mouths;
};

// Deterministic CV_32FC1 map, baseline 1. Smooth elliptical regions combine by
// maximum, so overlap never increases the configured peak. Missing features
// receive weaker geometric priors within a detected face (one prior per eye).
// Weights are sanitized to [1, 15]; non-finite values use their defaults.
cv::Mat build_facial_saliency(cv::Size size,
    const std::vector<FacialRegions>& faces,
    const FacialSaliencyConfig& config = {});

// Input is CV_8UC3 RGB. Frontal-face Haar detection gates eye detection and
// optional smile-based mouth refinement; a smile model is not a mouth detector.
cv::Mat compute_facial_saliency(const cv::Mat& rgb,
    const FacialSaliencyConfig& config = {});

} // namespace vinylizer
