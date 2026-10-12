#include "preprocess/facial_saliency.h"
#include "core/saliency_weights.h"
#include <opencv2/objdetect.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(float actual, float expected, const char* message) {
    require(std::isfinite(actual) && std::abs(actual - expected) < 1e-4f, message);
}

void test_maps() {
    using namespace vinylizer;
    const cv::Size size(256, 256);
    auto neutral = build_facial_saliency(size, {});
    require(cv::countNonZero(neutral != 1.0f) == 0, "No faces must produce neutral weights");
    require(build_facial_saliency({}, {}).empty(), "Empty canvas must be supported");
    FacialRegions face{{40, 32, 160, 192}, {{68, 92, 32, 24}, {148, 92, 32, 24}}, {{96, 170, 48, 24}}};
    auto map = build_facial_saliency(size, {face});
    near(map.at<float>(104, 84), 12.0f, "Detected eye peak");
    near(map.at<float>(182, 120), 10.0f, "Detected mouth peak");
    near(map.at<float>(128, 120), 3.0f, "Face center weight");
    near(map.at<float>(0, 0), 1.0f, "Distant background stays neutral");
    require(map.at<float>(104, 66) > 1.0f && map.at<float>(104, 66) < 12.0f,
            "Eye weighting must have a smooth fringe");
    auto duplicate = build_facial_saliency(size, {face, face});
    near(static_cast<float>(cv::norm(map, duplicate, cv::NORM_INF)), 0.0f,
         "Overlapping detections must not compound weights");

    auto fallback = build_facial_saliency(size, {{{40, 32, 160, 192}, {}, {}}});
    require(fallback.at<float>(105, 84) > 3.0f && fallback.at<float>(105, 84) <= 7.0f,
            "Missing eye should receive a weaker prior");
    require(fallback.at<float>(180, 120) > 3.0f && fallback.at<float>(180, 120) <= 6.0f,
            "Neutral mouth should receive a weaker prior");
    FacialRegions one_eye = face;
    one_eye.eyes.pop_back();
    auto partial = build_facial_saliency(size, {one_eye});
    near(partial.at<float>(104, 84), 12.0f, "Keep a detected eye's stronger weight");
    require(partial.at<float>(105, 156) > 3.0f && partial.at<float>(105, 156) <= 7.0f,
            "Missing opposite eye retains its prior");

    FacialRegions clipped{{-20, -20, 60, 60}, {{-10, -10, 20, 20}}, {{500, 500, 40, 20}}};
    auto clipped_map = build_facial_saliency({16, 16}, {clipped});
    require(cv::checkRange(clipped_map), "Clipped/invalid rectangles stay finite");
    FacialSaliencyConfig disabled{1, 1, 1, 1, 1};
    require(cv::countNonZero(build_facial_saliency(size, {face}, disabled) != 1.0f) == 0,
            "All weights one disables enhancement");
}

void test_normalization() {
    using namespace vinylizer;
    cv::Mat mask = (cv::Mat_<unsigned char>(2, 3) << 255, 255, 0, 255, 0, 0);
    auto empty = build_saliency_weights({}, mask);
    require(cv::countNonZero(empty.rgb_loss != 1.0f) == 0, "Absent map preserves baseline");
    auto constant = build_saliency_weights(cv::Mat(2, 3, CV_32F, cv::Scalar(7)), mask);
    require(cv::countNonZero(constant.rgb_loss != 1.0f) == 0, "Constant map preserves loss scale");
    cv::Mat priority = (cv::Mat_<float>(2, 3) << 12, 1, 15, 3, 15, 15);
    auto weights = build_saliency_weights(priority, mask);
    near(static_cast<float>(cv::mean(weights.rgb_loss, mask)[0]), 1.0f, "Opaque mean must be one");
    near(weights.rgb_loss.at<float>(0, 0) / weights.rgb_loss.at<float>(0, 1), 12.0f,
         "Normalization must preserve relative feature importance");
    near(weights.priority.at<float>(0, 2), 1.0f, "Transparent detection must be neutral");
    near(weights.rgb_loss.at<float>(0, 2), 1.0f, "Transparent weights must not alter normalization");
    auto transparent = build_saliency_weights(priority, cv::Mat::zeros(2, 3, CV_8U));
    require(cv::countNonZero(transparent.rgb_loss != 1.0f) == 0, "All transparent input stays finite/neutral");
    priority.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    priority.at<float>(0, 1) = std::numeric_limits<float>::infinity();
    priority.at<float>(1, 0) = -5;
    auto bad = build_saliency_weights(priority, mask);
    require(cv::checkRange(bad.rgb_loss), "Invalid weights cannot propagate NaNs");
    auto resized = build_saliency_weights(cv::Mat(1, 1, CV_32F, cv::Scalar(3)), mask);
    require(resized.rgb_loss.size() == mask.size() && resized.rgb_loss.isContinuous(),
            "Resized GPU weight upload must be contiguous and match target");
    near(saliency_radius_scale(1), 1, "Neutral shapes retain original scale");
    require(saliency_radius_scale(12) < saliency_radius_scale(3), "Eyes get finer primitives than face");
}

void test_assets(const std::filesystem::path& directory) {
    for (const char* name : {"haarcascade_frontalface_default.xml", "haarcascade_eye.xml", "haarcascade_smile.xml"}) {
        std::ifstream file(directory / name, std::ios::binary);
        require(file.good(), "Deployed cascade asset must exist");
        std::string xml((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        cv::FileStorage storage(xml, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        cv::CascadeClassifier classifier;
        require(storage.isOpened() && classifier.read(storage.getFirstTopLevelNode()) && !classifier.empty(),
                "Deployed cascade must parse using the production Unicode-safe loading method");
    }
    auto blank = vinylizer::compute_facial_saliency(cv::Mat::zeros(128, 128, CV_8UC3));
    require(cv::countNonZero(blank != 1.0f) == 0, "Blank image must not invent facial features");
    auto tiny = vinylizer::compute_facial_saliency(cv::Mat::zeros(8, 8, CV_8UC3));
    require(cv::countNonZero(tiny != 1.0f) == 0, "Tiny image stays neutral");
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Pass the deployed cascade directory");
        test_maps();
        test_normalization();
        test_assets(std::filesystem::u8path(argv[1]));
        std::cout << "PASS: facial maps, feature fallbacks, masked normalization, assets\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
