#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vinylizer {
struct ResidualSeed {
    int x, y;
    float rx, ry, angle;
};

// Local second moments set ellipse axes; suppress each chosen footprint before
// selecting the next. A lazy max heap avoids an image scan per replacement.
inline std::vector<ResidualSeed> select_residual_seeds(
    const float* error, int width, int height, int count,
    float radius_lo, float radius_hi, const float* radius_scale = nullptr)
{
    if (!error || width <= 0 || height <= 0 || count < 0
        || width > std::numeric_limits<int>::max() / height
        || !std::isfinite(radius_lo) || !std::isfinite(radius_hi)
        || radius_lo <= 0.0f || radius_hi < radius_lo)
        throw std::invalid_argument("Invalid residual placement inputs");
    const int pixels = width * height;
    using Entry = std::pair<float, int>;
    std::vector<float> score(pixels);
    std::vector<Entry> entries;
    entries.reserve(pixels);
    for (int i = 0; i < pixels; ++i) {
        if (std::isfinite(error[i]) && error[i] > 0.0f) {
            score[i] = error[i];
            entries.emplace_back(score[i], i);
        }
    }
    std::priority_queue<Entry> heap(std::less<Entry>{}, std::move(entries));
    std::vector<ResidualSeed> seeds;
    seeds.reserve(std::min(count, pixels));
    const int reach = static_cast<int>(std::ceil(std::min(radius_hi,
        static_cast<float>(std::max(width, height)))));
    while (static_cast<int>(seeds.size()) < count && !heap.empty()) {
        const auto [peak, index] = heap.top();
        heap.pop();
        if (peak != score[index] || peak <= 0.0f) continue;
        const int x = index % width, y = index / width;
        double mass = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        const int x0 = std::max(0, x - reach), x1 = std::min(width - 1, x + reach);
        const int y0 = std::max(0, y - reach), y1 = std::min(height - 1, y + reach);
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                const double dx = xx - x, dy = yy - y;
                if (dx * dx + dy * dy > double(reach) * reach) continue;
                const double w = score[yy * width + xx];
                mass += w; sx += w * dx; sy += w * dy;
                sxx += w * dx * dx; syy += w * dy * dy; sxy += w * dx * dy;
            }
        }
        const double mx = sx / mass, my = sy / mass;
        const double vx = std::max(0.0, sxx / mass - mx * mx);
        const double vy = std::max(0.0, syy / mass - my * my);
        const double cov = sxy / mass - mx * my;
        const double delta = std::hypot(vx - vy, 2.0 * cov);
        const float major = static_cast<float>(2.0 * std::sqrt(std::max(0.0, (vx + vy + delta) * 0.5)));
        const float minor = static_cast<float>(2.0 * std::sqrt(std::max(0.0, (vx + vy - delta) * 0.5)));
        const float scale = radius_scale && std::isfinite(radius_scale[index])
            ? std::clamp(radius_scale[index], 0.1f, 1.0f) : 1.0f;
        const float rx = std::max(1.0f, std::clamp(major, radius_lo, radius_hi) * scale);
        const float ry = std::max(1.0f, std::clamp(minor, radius_lo, radius_hi) * scale);
        const float radians = static_cast<float>(0.5 * std::atan2(2.0 * cov, vx - vy));
        seeds.push_back({x, y, rx, ry, radians * (180.0f / 3.14159265358979323846f)});
        const float cosine = std::cos(radians), sine = std::sin(radians);
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                const float dx = static_cast<float>(xx - x), dy = static_cast<float>(yy - y);
                const float u = (dx * cosine + dy * sine) / rx;
                const float v = (-dx * sine + dy * cosine) / ry;
                const float distance = u * u + v * v;
                const int i = yy * width + xx;
                if (distance >= 1.0f || score[i] <= 0.0f) continue;
                score[i] *= distance;
                if (score[i] > 0.0f) heap.emplace(score[i], i);
            }
        }
    }
    return seeds;
}
} // namespace vinylizer
