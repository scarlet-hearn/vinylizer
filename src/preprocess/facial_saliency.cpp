#include "preprocess/facial_saliency.h"
#include "common/path_utils.h"
#include "common/logger.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <utility>

namespace vinylizer {
namespace {

float bounded_weight(float value, float fallback) {
    return std::isfinite(value) ? std::clamp(value, 1.0f, 15.0f) : fallback;
}

cv::Rect relative_rect(const cv::Rect& face, float x, float y, float w, float h) {
    return cv::Rect(face.x + cvRound(x * face.width),
                    face.y + cvRound(y * face.height),
                    std::max(1, cvRound(w * face.width)),
                    std::max(1, cvRound(h * face.height))) & face;
}

void paint_ellipse(cv::Mat& map, const cv::Rect& region, float peak,
                   float width_scale, float height_scale, const cv::Rect& clip) {
    if (region.empty() || peak <= 1.0f) return;
    const float cx = region.x + 0.5f * region.width;
    const float cy = region.y + 0.5f * region.height;
    const float rx = std::max(0.5f, width_scale * region.width);
    const float ry = std::max(0.5f, height_scale * region.height);
    const int x0 = std::max(clip.x, static_cast<int>(std::floor(cx - rx)));
    const int y0 = std::max(clip.y, static_cast<int>(std::floor(cy - ry)));
    const int x1 = std::min(clip.x + clip.width, static_cast<int>(std::ceil(cx + rx)));
    const int y1 = std::min(clip.y + clip.height, static_cast<int>(std::ceil(cy + ry)));
    for (int y = y0; y < y1; ++y) {
        float* row = map.ptr<float>(y);
        const float dy = (y + 0.5f - cy) / ry;
        for (int x = x0; x < x1; ++x) {
            const float dx = (x + 0.5f - cx) / rx;
            const float radius = std::sqrt(dx * dx + dy * dy);
            if (radius >= 1.0f) continue;
            // Flat inner core and a C1-continuous feather to baseline.
            const float t = std::clamp((radius - 0.4f) / 0.6f, 0.0f, 1.0f);
            const float blend = 1.0f - t * t * (3.0f - 2.0f * t);
            row[x] = std::max(row[x], 1.0f + (peak - 1.0f) * blend);
        }
    }
}

std::filesystem::path executable_directory() {
#ifdef _WIN32
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                           static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) return {};
    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
#elif defined(__linux__)
    std::error_code error;
    const auto path = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::filesystem::path{} : path.parent_path();
#else
    return {};
#endif
}

std::vector<std::filesystem::path> cascade_directories() {
    std::vector<std::filesystem::path> directories;
    const auto executable = executable_directory();
    if (!executable.empty()) directories.push_back(executable / "assets" / "haarcascades");
#ifdef _WIN32
    const DWORD length = GetEnvironmentVariableW(L"VINYLIZER_CASCADE_DIR", nullptr, 0);
    if (length > 0) {
        std::wstring value(length, L'\0');
        const DWORD copied = GetEnvironmentVariableW(L"VINYLIZER_CASCADE_DIR", value.data(), length);
        if (copied > 0 && copied < length) {
            value.resize(copied);
            directories.emplace_back(value);
        }
    }
#else
    if (const char* directory = std::getenv("VINYLIZER_CASCADE_DIR"))
        directories.emplace_back(directory);
#endif
    std::error_code error;
    const auto current = std::filesystem::current_path(error);
    if (!error) directories.push_back(current / "assets" / "haarcascades");
    return directories;
}

bool read_cascade(cv::CascadeClassifier& classifier, const std::filesystem::path& path) {
    // OpenCV's narrow filename loader does not reliably support Unicode Windows
    // paths. Read through filesystem::path, then parse the XML from memory.
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    const std::string contents((std::istreambuf_iterator<char>(stream)),
                                std::istreambuf_iterator<char>());
    if (contents.empty() || stream.bad()) return false;
    try {
        cv::FileStorage storage(contents, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        return storage.isOpened() && classifier.read(storage.getFirstTopLevelNode())
            && !classifier.empty();
    } catch (const cv::Exception&) {
        return false;
    }
}

bool load_cascade(cv::CascadeClassifier& classifier, const char* filename) {
    for (const auto& directory : cascade_directories()) {
        if (read_cascade(classifier, directory / filename)) return true;
    }
    // Honor OpenCV's own sample-data configuration as a final development fallback.
    for (const auto& relative : {std::string(filename), std::string("haarcascades/") + filename}) {
        try {
            const auto path = cv::samples::findFile(relative, false, true);
            if (!path.empty() && read_cascade(classifier, std::filesystem::u8path(path))) return true;
        } catch (const cv::Exception&) {
            // An unavailable sample-data directory must not break preprocessing.
        }
    }
    return false;
}

struct CascadeModels {
    cv::CascadeClassifier face;
    cv::CascadeClassifier eyes;
    cv::CascadeClassifier smile;

    CascadeModels() {
        const bool has_face = load_cascade(face, "haarcascade_frontalface_default.xml");
        const bool has_eyes = load_cascade(eyes, "haarcascade_eye.xml");
        const bool has_smile = load_cascade(smile, "haarcascade_smile.xml");
        if (!has_face) {
            VIN_WARN("Face cascade unavailable: facial weighting disabled; expected assets/haarcascades beside the executable or VINYLIZER_CASCADE_DIR");
        } else if (!has_eyes || !has_smile) {
            VIN_WARN("Facial cascades: eyes=%s, smile refinement=%s; missing features use weaker geometric priors",
                     has_eyes ? "loaded" : "missing", has_smile ? "loaded" : "missing");
        }
    }
};

std::vector<cv::Rect> detect_in_roi(cv::CascadeClassifier& classifier,
    const cv::Mat& gray, const cv::Rect& roi, cv::Size minimum, cv::Size maximum,
    int neighbors) {
    std::vector<cv::Rect> regions;
    if (classifier.empty() || roi.empty()) return regions;
    const cv::Size window = classifier.getOriginalWindowSize();
    minimum.width = std::max(minimum.width, window.width);
    minimum.height = std::max(minimum.height, window.height);
    maximum.width = std::min(maximum.width, roi.width);
    maximum.height = std::min(maximum.height, roi.height);
    if (maximum.width < minimum.width || maximum.height < minimum.height) return regions;
    classifier.detectMultiScale(gray(roi), regions, 1.08, neighbors, 0, minimum, maximum);
    for (auto& region : regions) region += roi.tl();
    return regions;
}

cv::Rect scaled_rect(const cv::Rect& rect, double sx, double sy, cv::Size size) {
    const int left = cvRound(rect.x * sx);
    const int top = cvRound(rect.y * sy);
    return cv::Rect(left, top, std::max(1, cvRound((rect.x + rect.width) * sx) - left),
                    std::max(1, cvRound((rect.y + rect.height) * sy) - top))
        & cv::Rect(0, 0, size.width, size.height);
}

} // namespace

cv::Mat build_facial_saliency(cv::Size size, const std::vector<FacialRegions>& faces,
                              const FacialSaliencyConfig& config) {
    if (size.width <= 0 || size.height <= 0) return {};
    cv::Mat map(size, CV_32FC1, cv::Scalar(1.0f));
    const cv::Rect image(0, 0, size.width, size.height);
    const float face_weight = bounded_weight(config.face_weight, 3.0f);
    const float eye_weight = bounded_weight(config.eye_weight, 12.0f);
    const float mouth_weight = bounded_weight(config.mouth_weight, 10.0f);
    const float fallback_eye = std::min(eye_weight, bounded_weight(config.fallback_eye_weight, 7.0f));
    const float fallback_mouth = std::min(mouth_weight, bounded_weight(config.fallback_mouth_weight, 6.0f));

    for (const auto& regions : faces) {
        const cv::Rect face = regions.face & image;
        if (face.empty()) continue;
        paint_ellipse(map, face, face_weight, 0.58f, 0.62f, image);
        std::array<bool, 2> eye_found{false, false};
        for (const auto& detected : regions.eyes) {
            const cv::Rect eye = detected & face;
            if (eye.empty()) continue;
            const float cx = eye.x + 0.5f * eye.width;
            const float cy = eye.y + 0.5f * eye.height;
            if (cy < face.y + 0.12f * face.height || cy > face.y + 0.62f * face.height) continue;
            const int side = cx < face.x + 0.5f * face.width ? 0 : 1;
            eye_found[side] = true;
            paint_ellipse(map, eye, eye_weight, 0.65f, 0.55f, face);
        }
        if (!eye_found[0]) paint_ellipse(map, relative_rect(face, 0.14f, 0.27f, 0.27f, 0.22f),
                                         fallback_eye, 0.65f, 0.6f, face);
        if (!eye_found[1]) paint_ellipse(map, relative_rect(face, 0.59f, 0.27f, 0.27f, 0.22f),
                                         fallback_eye, 0.65f, 0.6f, face);
        bool mouth_found = false;
        for (const auto& detected : regions.mouths) {
            const cv::Rect mouth = detected & face;
            if (mouth.empty()) continue;
            const float cy = mouth.y + 0.5f * mouth.height;
            if (cy < face.y + 0.55f * face.height) continue;
            mouth_found = true;
            paint_ellipse(map, mouth, mouth_weight, 0.6f, 0.6f, face);
        }
        if (!mouth_found) paint_ellipse(map, relative_rect(face, 0.28f, 0.67f, 0.44f, 0.20f),
                                        fallback_mouth, 0.6f, 0.65f, face);
    }
    return map;
}

cv::Mat compute_facial_saliency(const cv::Mat& rgb, const FacialSaliencyConfig& config) {
    if (rgb.empty()) return {};
    CV_Assert(rgb.type() == CV_8UC3);
    if (rgb.cols < 30 || rgb.rows < 30) return build_facial_saliency(rgb.size(), {}, config);
    // Each preprocessing worker gets its own detector cache; classifiers are mutable.
    static thread_local CascadeModels models;
    if (models.face.empty()) return build_facial_saliency(rgb.size(), {}, config);

    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    const double scale = std::min(1.0, 960.0 / std::max(rgb.cols, rgb.rows));
    if (scale < 1.0) {
        const cv::Size size(std::max(1, cvRound(gray.cols * scale)),
                            std::max(1, cvRound(gray.rows * scale)));
        cv::resize(gray, gray, size, 0, 0, cv::INTER_AREA);
    }
    if (gray.cols < 30 || gray.rows < 30) return build_facial_saliency(rgb.size(), {}, config);
    cv::equalizeHist(gray, gray);
    std::vector<cv::Rect> faces;
    models.face.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(30, 30));
    const cv::Rect image(0, 0, gray.cols, gray.rows);
    const double sx = static_cast<double>(rgb.cols) / gray.cols;
    const double sy = static_cast<double>(rgb.rows) / gray.rows;
    std::vector<FacialRegions> regions;
    int eye_count = 0, mouth_count = 0;

    for (const auto& candidate : faces) {
        const cv::Rect face = candidate & image;
        if (face.empty()) continue;
        FacialRegions features;
        features.face = scaled_rect(face, sx, sy, rgb.size());
        // Search the upper face only. Retain one anatomically plausible eye per side.
        const auto eyes = detect_in_roi(models.eyes, gray,
            relative_rect(face, 0.05f, 0.12f, 0.9f, 0.52f),
            cv::Size(cvRound(face.width * 0.12f), cvRound(face.height * 0.10f)),
            cv::Size(cvRound(face.width * 0.45f), cvRound(face.height * 0.40f)), 4);
        std::array<cv::Rect, 2> best_eyes;
        std::array<float, 2> scores{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        for (const auto& eye : eyes) {
            const float x = (eye.x + 0.5f * eye.width - face.x) / face.width;
            const float y = (eye.y + 0.5f * eye.height - face.y) / face.height;
            if (x < 0.12f || x > 0.88f || y < 0.20f || y > 0.55f) continue;
            const int side = x < 0.5f ? 0 : 1;
            const float dx = x - (side == 0 ? 0.28f : 0.72f);
            const float dy = y - 0.38f;
            const float score = dx * dx + dy * dy;
            if (score < scores[side]) { scores[side] = score; best_eyes[side] = eye; }
        }
        for (const auto& eye : best_eyes) {
            if (!eye.empty()) { features.eyes.push_back(scaled_rect(eye, sx, sy, rgb.size())); ++eye_count; }
        }

        // Smile detections refine mouth placement when present. Neutral or closed
        // mouths still receive the lower-confidence geometric mouth prior.
        const auto smiles = detect_in_roi(models.smile, gray,
            relative_rect(face, 0.12f, 0.55f, 0.76f, 0.40f),
            cv::Size(cvRound(face.width * 0.25f), cvRound(face.height * 0.10f)),
            cv::Size(cvRound(face.width * 0.70f), cvRound(face.height * 0.35f)), 12);
        cv::Rect best_mouth;
        float mouth_score = std::numeric_limits<float>::max();
        for (const auto& mouth : smiles) {
            const float x = (mouth.x + 0.5f * mouth.width - face.x) / face.width;
            const float y = (mouth.y + 0.5f * mouth.height - face.y) / face.height;
            if (x < 0.30f || x > 0.70f || y < 0.62f || y > 0.90f) continue;
            const float score = (x - 0.5f) * (x - 0.5f) + (y - 0.77f) * (y - 0.77f);
            if (score < mouth_score) { mouth_score = score; best_mouth = mouth; }
        }
        if (!best_mouth.empty()) {
            features.mouths.push_back(scaled_rect(best_mouth, sx, sy, rgb.size()));
            ++mouth_count;
        }
        regions.push_back(std::move(features));
    }
    VIN_INFO("Facial saliency: %d faces, %d detected eyes, %d smile-based mouth refinements; missing features use geometric priors",
             static_cast<int>(regions.size()), eye_count, mouth_count);
    return build_facial_saliency(rgb.size(), regions, config);
}

} // namespace vinylizer
