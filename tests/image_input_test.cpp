#include "preprocess/image_input.h"
#include <iostream>
#include <stdexcept>

void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
int main() {
    try {
        using namespace vinylizer;
        cv::Mat large(2160, 3840, CV_8UC3, cv::Scalar(40, 80, 120));
        auto small = resize_input_image(normalize_input_image(large), 512);
        require(small.cols == 512 && small.rows == 288, "4K input did not downsample to configured resolution");
        auto tiny = resize_input_image(cv::Mat::zeros(1, 1000, CV_8UC3), 128);
        require(tiny.rows == 1 && tiny.cols == 128, "Thin input got a zero dimension");
        cv::Mat gray16(2, 2, CV_16UC1, cv::Scalar(65535));
        auto normal = normalize_input_image(gray16);
        require(normal.type() == CV_8UC3 && normal.at<cv::Vec3b>(0, 0)[0] == 255, "16-bit grayscale normalization failed");
        cv::Mat rgba(1, 2, CV_8UC4);
        rgba.at<cv::Vec4b>(0, 0) = {0, 0, 255, 255};
        rgba.at<cv::Vec4b>(0, 1) = {255, 0, 0, 0};
        auto resized = resize_input_image(rgba, 1);
        const auto pixel = resized.at<cv::Vec4b>(0, 0);
        require(pixel[0] == 0 && pixel[2] == 255 && pixel[3] >= 127 && pixel[3] <= 128,
                "Hidden blue bled into red, or intermediate alpha was lost");
        cv::Mat transparent(8, 8, CV_8UC4, cv::Scalar(255, 255, 255, 0));
        auto empty = resize_input_image(transparent, 2);
        require(cv::countNonZero(empty.reshape(1)) == 0, "Fully transparent resize must stay finite and zero");
        std::cout << "PASS: resolution, aspect ratio, bit depth and premultiplied alpha resize\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
