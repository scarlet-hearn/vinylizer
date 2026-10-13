#include "core/optimizer.h"
#include "core/saliency_weights.h"
#include "core/alpha_target.h"
#include "core/residual_placement.h"
#include "cuda/check.h"
#include "common/logger.h"
#include "common/canvas.h"

#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace vinylizer {

// ================================================================
// OptParams
// ================================================================

void OptParams::init(int n) {
    N = n;
    cx.resize(n); cy.resize(n); rx.resize(n); ry.resize(n); angle.resize(n);
    color_r.resize(n); color_g.resize(n); color_b.resize(n); color_a.resize(n);
}

ShapeParams OptParams::get_shape(int i) const {
    ShapeParams sp;
    sp.cx = cx[i]; sp.cy = cy[i]; sp.rx = rx[i]; sp.ry = ry[i];
    sp.angle = angle[i];
    sp.color_r = color_r[i]; sp.color_g = color_g[i]; sp.color_b = color_b[i];
    sp.opacity = color_a[i] / 255.0f;
    return sp;
}

void OptParams::set_shape(int i, const ShapeParams& sp) {
    cx[i] = sp.cx; cy[i] = sp.cy; rx[i] = sp.rx; ry[i] = sp.ry;
    angle[i] = sp.angle;
    color_r[i] = sp.color_r; color_g[i] = sp.color_g;
    color_b[i] = sp.color_b; color_a[i] = sp.opacity * 255.0f;
}

// ================================================================
// GradientOptimizer
// ================================================================

GradientOptimizer::GradientOptimizer(const OptimizerConfig& config)
    : config_(config) {}

GradientOptimizer::~GradientOptimizer() {
    // Synchronize before releasing memory referenced by async copies, including on exceptions.
    try { renderer_.sync_stream(); } catch (...) {}
    if (d_grad_history_) cudaFree(d_grad_history_);
    if (viz_pinned_) cudaFreeHost(viz_pinned_);
    if (snap_pinned_) cudaFreeHost(snap_pinned_);
    if (viz_event_) cudaEventDestroy(viz_event_);
    if (snap_event_) cudaEventDestroy(snap_event_);
}

std::vector<Shape> GradientOptimizer::optimize(
    const cv::Mat& target_rgb, const cv::Mat& mask,
    const cv::Mat& edge_map, const cv::Mat& saliency_map,
    const cv::Mat& dark_mask, const cv::Mat& posterized,
    const std::vector<std::tuple<int,int,int>>& palette,
    const cv::Mat& grad_dir, const Color& bg_color, const cv::Mat& target_alpha)
{
    target_rgb_ = target_rgb.clone();
    mask_ = mask.clone();
    edge_map_ = edge_map.clone();
    saliency_map_ = saliency_map.clone();
    dark_mask_ = dark_mask.clone();
    palette_ = palette;
    stop_requested_ = false;

    // Compute optimization resolution
    int H_orig = target_rgb.rows, W_orig = target_rgb.cols;
    int max_dim = std::max(H_orig, W_orig);
    scale_ = std::min(1.0f, static_cast<float>(config_.opt_resolution) / max_dim);
    H_opt_ = std::max(1, static_cast<int>(H_orig * scale_));
    W_opt_ = std::max(1, static_cast<int>(W_orig * scale_));

    VIN_INFO("Optimizer: %dx%d -> %dx%d (scale=%.3f)",
             W_orig, H_orig, W_opt_, H_opt_, scale_);

    // Resize target and mask
    cv::resize(target_rgb, target_small_, {W_opt_, H_opt_}, 0, 0, cv::INTER_AREA);
    cv::Mat mask_u8;
    mask.convertTo(mask_u8, CV_8U, 255);
    cv::resize(mask_u8, mask_small_, {W_opt_, H_opt_}, 0, 0, cv::INTER_NEAREST);
    mask_small_ = mask_small_ > 127;

    cv::Mat alpha_small;
    if (!target_alpha.empty()) {
        CV_Assert(target_alpha.type() == CV_8UC1 && target_alpha.size() == target_rgb.size());
        cv::resize(target_alpha, alpha_small, {W_opt_, H_opt_}, 0, 0, cv::INTER_AREA);
        mask_small_ = alpha_small > 0;
    }

    auto facial_weights = build_saliency_weights(saliency_map_, mask_small_);
    saliency_small_ = facial_weights.priority;
    rgb_loss_weights_ = facial_weights.rgb_loss;

    // Resize posterized
    if (!posterized.empty()) {
        cv::resize(posterized, posterized_small_, {W_opt_, H_opt_}, 0, 0, cv::INTER_NEAREST);
    }

    // Initialize renderer
    renderer_.init(W_opt_, H_opt_, config_.max_shapes);
    bg_color_ = bg_color;
    renderer_.set_bg_color(static_cast<float>(bg_color.r), static_cast<float>(bg_color.g), static_cast<float>(bg_color.b));

    // Set target on GPU (RGB + mask + target_alpha)
    std::vector<float> h_target(H_opt_ * W_opt_ * 3);
    std::vector<float> h_mask_gpu(H_opt_ * W_opt_);
    std::vector<float> h_target_alpha(H_opt_ * W_opt_);
    for (int y = 0; y < H_opt_; ++y) {
        for (int x = 0; x < W_opt_; ++x) {
            int idx = y * W_opt_ + x;
            bool opaque = mask_small_.at<uint8_t>(y, x) > 0;
            if (opaque) {
                auto& c = target_small_.at<cv::Vec3b>(y, x);
                // Renderer outputs color composited over bg, in linear-light space.
                const float alpha = alpha_small.empty() ? 1.0f : alpha_small.at<uint8_t>(y, x) / 255.0f;
                h_target[idx * 3 + 0] = composite_target_color(c[0], bg_color.r, alpha);
                h_target[idx * 3 + 1] = composite_target_color(c[1], bg_color.g, alpha);
                h_target[idx * 3 + 2] = composite_target_color(c[2], bg_color.b, alpha);
            } else {
                // Transparent pixels: target RGB = bg_color (so rendered bg matches)
                h_target[idx * 3 + 0] = static_cast<float>(bg_color.r);
                h_target[idx * 3 + 1] = static_cast<float>(bg_color.g);
                h_target[idx * 3 + 2] = static_cast<float>(bg_color.b);
            }
            h_mask_gpu[idx] = opaque ? 1.0f : 0.0f;
            h_target_alpha[idx] = alpha_small.empty() ? (opaque ? 255.0f : 0.0f)
                : static_cast<float>(alpha_small.at<uint8_t>(y, x));
        }
    }
    renderer_.set_target(h_target.data(), h_mask_gpu.data(), h_target_alpha.data(),
                         rgb_loss_weights_.ptr<float>());
    loss_target_small_ = cv::Mat(H_opt_, W_opt_, CV_32FC3, h_target.data()).clone();
    target_alpha_small_ = cv::Mat(H_opt_, W_opt_, CV_32FC1, h_target_alpha.data()).clone();

    // Initialize shapes
    initialize_shapes();

    // Run optimization
    gradient_optimize();

    // Convert back to shapes (best params already restored inside gradient_optimize)
    return params_to_shapes();
}

// ================================================================
// Initialization
// ================================================================

cv::Mat GradientOptimizer::build_importance_map() {
    cv::Mat importance = cv::Mat::ones(H_opt_, W_opt_, CV_32FC1);

    if (!edge_map_.empty()) {
        cv::Mat edge_small;
        cv::resize(edge_map_, edge_small, {W_opt_, H_opt_}, 0, 0, cv::INTER_LINEAR);
        cv::dilate(edge_small, edge_small, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
        cv::Mat weighted;
        edge_small.convertTo(weighted, CV_32FC1, 3.0 / 255.0, 1.0);
        importance = importance.mul(weighted);
    }

    importance = importance.mul(saliency_small_);

    if (!dark_mask_.empty()) {
        cv::Mat dark_small;
        cv::resize(dark_mask_, dark_small, {W_opt_, H_opt_}, 0, 0, cv::INTER_LINEAR);
        cv::Mat weighted;
        dark_small.convertTo(weighted, CV_32FC1, 1.5 / 255.0, 1.0);
        importance = importance.mul(weighted);
    }

    importance.setTo(0, ~mask_small_);
    double total = cv::sum(importance)[0];
    if (total > 0) importance /= total;

    return importance;
}

void GradientOptimizer::initialize_shapes() {
    int N = config_.max_shapes;
    params_.init(N);

    cv::Mat importance = build_importance_map();

    int long_edge = std::max(W_opt_, H_opt_);
    float size_lo = long_edge * (config_.init_radius_lo / 100.0f);
    float size_hi = long_edge * (config_.init_radius_hi / 100.0f);
    std::uniform_real_distribution<float> dist_size(size_lo, size_hi);
    std::uniform_real_distribution<float> dist_angle(0.0f, 360.0f);

    // Weighted sampling from importance map
    cv::Mat flat_importance = importance.reshape(1, 1);
    std::vector<float> probs(flat_importance.cols);
    flat_importance.copyTo(cv::Mat(1, flat_importance.cols, CV_32FC1, probs.data()));

    double prob_sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (prob_sum > 0) {
        std::discrete_distribution<int> dist_pos(probs.begin(), probs.end());
        for (int i = 0; i < N; ++i) {
            int idx = dist_pos(rng_);
            int y = idx / W_opt_, x = idx % W_opt_;
            params_.cx[i] = static_cast<float>(x);
            params_.cy[i] = static_cast<float>(y);
            float detail_scale = saliency_radius_scale(saliency_small_.at<float>(y, x));
            params_.rx[i] = std::max(1.0f, dist_size(rng_) * detail_scale);
            params_.ry[i] = std::max(1.0f, dist_size(rng_) * detail_scale);
            params_.angle[i] = dist_angle(rng_);
            auto [r, g, b] = get_local_color(static_cast<float>(x), static_cast<float>(y));
            params_.color_r[i] = r;
            params_.color_g[i] = g;
            params_.color_b[i] = b;
            params_.color_a[i] = 255.0f;
        }
    }

    VIN_INFO("Initialized %d shapes", N);
}

std::tuple<int,int,int> GradientOptimizer::get_local_color(float x, float y) {
    int xi = std::clamp(static_cast<int>(x), 0, W_opt_ - 1);
    int yi = std::clamp(static_cast<int>(y), 0, H_opt_ - 1);
    int r = std::max(1, static_cast<int>(std::lround(
        5.0f * saliency_radius_scale(saliency_small_.at<float>(yi, xi)))));
    int x0 = std::max(0, static_cast<int>(x) - r);
    int x1 = std::min(W_opt_, static_cast<int>(x) + r + 1);
    int y0 = std::max(0, static_cast<int>(y) - r);
    int y1 = std::min(H_opt_, static_cast<int>(y) + r + 1);

    cv::Mat local = target_small_(cv::Range(y0, y1), cv::Range(x0, x1));
    cv::Mat local_mask = mask_small_(cv::Range(y0, y1), cv::Range(x0, x1));

    cv::Scalar mean = cv::mean(local, local_mask);
    return {static_cast<int>(mean[0]), static_cast<int>(mean[1]), static_cast<int>(mean[2])};
}

std::tuple<int,int,int> GradientOptimizer::get_posterized_color(float x, float y) {
    if (!posterized_small_.empty() && !palette_.empty()) {
        int yi = std::clamp(static_cast<int>(y), 0, H_opt_ - 1);
        int xi = std::clamp(static_cast<int>(x), 0, W_opt_ - 1);
        int label = posterized_small_.at<uint8_t>(yi, xi);
        if (label < static_cast<int>(palette_.size())) {
            return palette_[label];
        }
    }
    return get_local_color(x, y);
}

// ================================================================
// Single iteration
// ================================================================

float GradientOptimizer::run_single_iter(int iter, float* d_grad_history, int write_idx)
{
    int N = params_.N;

    // Forward + backward on GPU (optionally writes gradient norms to history)
    float mse = renderer_.render_forward_backward(N, d_grad_history, write_idx);
    capture_best(mse);

    // Learning rate schedule (cosine)
    float progress = static_cast<float>(iter) / std::max(config_.total_iters - 1, 1);
    float lr = config_.lr_end + (config_.lr_start - config_.lr_end) *
               0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));

    // Fused GPU Adam: gradient clip + Adam update + param clamp + STE quantize
    adam_step_++;
    renderer_.fused_adam_update(
        N, adam_step_, lr, 0.9f, 0.999f, 1e-8f,
        100.0f,   // max_grad_norm
        W_opt_, H_opt_);

    return mse;
}

void GradientOptimizer::publish_progress(float mse, int completed_iters) {
    if (!viz_first_frame_) {
        VIN_CUDA(cudaEventSynchronize(viz_event_));
        cv::Mat frame(H_opt_, W_opt_, CV_8UC3, viz_pinned_);
        std::lock_guard<std::mutex> lock(viz_mutex_);
        const int index = viz_write_idx_.load();
        canvas_rgb_[index] = frame.clone();
        viz_write_idx_.store(1 - index);
    }
    renderer_.async_copy_canvas(viz_pinned_, viz_event_);
    viz_first_frame_ = false;
    if (progress_cb_) progress_cb_(static_cast<float>(completed_iters) / config_.total_iters,
                                   mse, completed_iters);
}

// ================================================================
// Main optimization loop
// ================================================================

void GradientOptimizer::capture_best(float mse) {
    if (!std::isfinite(mse)) throw std::runtime_error("Non-finite optimization loss");
    if (!has_best_ || mse < best_mse_) {
        // Same stream: copy the evaluated parameters BEFORE Adam modifies them.
        renderer_.async_snapshot_params(snap_pinned_, snap_event_, params_.N);
        best_mse_ = mse;
        has_best_ = true;
    }
}

void GradientOptimizer::gradient_optimize() {
    const int N = params_.N;
    const int cycle_global = config_.reloc_cycle_global;
    const int cycle_local = config_.reloc_cycle_local;
    const int window = config_.reloc_stability_window;
    if (cycle_global <= 0 || cycle_local < 0 || window <= 0 || config_.total_iters <= 0
        || !std::isfinite(config_.reloc_max_fraction) || config_.reloc_max_fraction <= 0.0f
        || config_.reloc_max_fraction > 1.0f)
        throw std::invalid_argument("Invalid optimization cycle configuration");

    VIN_CUDA(cudaMalloc(&d_grad_history_, static_cast<size_t>(window) * N * sizeof(float)));
    VIN_CUDA(cudaMemset(d_grad_history_, 0, static_cast<size_t>(window) * N * sizeof(float)));
    VIN_CUDA(cudaMallocHost(&viz_pinned_, static_cast<size_t>(W_opt_) * H_opt_ * 3));
    VIN_CUDA(cudaEventCreate(&viz_event_));
    VIN_CUDA(cudaMallocHost(&snap_pinned_, static_cast<size_t>(N) * 9 * sizeof(float)));
    VIN_CUDA(cudaEventCreate(&snap_event_));
    adam_step_ = history_write_idx_ = history_count_ = reloc_total_ = 0;
    viz_first_frame_ = true;
    has_best_ = false;
    best_mse_ = 1e20f;
    renderer_.upload_params_gpu(params_.to_shape_params().data(), N);
    int completed_iters = 0, cycle = 0;

    auto global_step = [&] {
        const float mse = run_single_iter(completed_iters, d_grad_history_, history_write_idx_);
        history_write_idx_ = (history_write_idx_ + 1) % window;
        history_count_ = std::min(history_count_ + 1, window);
        publish_progress(mse, ++completed_iters);
    };
    while (completed_iters < config_.total_iters && !should_stop()) {
        ++cycle;
        for (int gi = 0; gi < cycle_global && completed_iters < config_.total_iters && !should_stop(); ++gi)
            global_step();
        if (should_stop()) break;
        const int local_iters = std::min(cycle_local, config_.total_iters - completed_iters);
        const bool last_cycle = config_.total_iters - completed_iters <= cycle_local;
        std::vector<int> candidates, indices;
        OptParams before_relocation;
        float previous_mse = 0.0f;
        if (!last_cycle && local_iters > 0) {
            // Refresh output and visibility at the same, post-update parameters.
            previous_mse = renderer_.render_forward_backward(N);
            capture_best(previous_mse);
            renderer_.compute_visibility(N);
            std::vector<float> visibility(N);
            renderer_.download_visibility(visibility.data(), N);
            auto shapes = params_.to_shape_params();
            renderer_.download_params_gpu(shapes.data(), N);
            for (int i = 0; i < N; ++i) params_.set_shape(i, shapes[i]);
            int useful = 0, useless = 0, unstable = 0;
            find_relocation_candidates(visibility, candidates, useful, useless, unstable);
            if (!candidates.empty()) {
                before_relocation = params_;
                indices = relocate_shapes(params_, candidates);
            }
        }
        if (!indices.empty()) {
            renderer_.upload_params_gpu(params_.to_shape_params().data(), N);
            renderer_.reset_adam_state_all_gpu(N);
            adam_step_ = history_write_idx_ = history_count_ = 0;
            std::vector<int> frozen(N, 1);
            for (int i : indices) frozen[i] = 0;
            renderer_.set_frozen_mask(frozen.data(), N);
            for (int li = 0; li < local_iters && !should_stop(); ++li) {
                const float mse = run_single_iter(completed_iters);
                publish_progress(mse, ++completed_iters);
            }
            std::fill(frozen.begin(), frozen.end(), 0);
            renderer_.set_frozen_mask(frozen.data(), N);
            const float proposed_mse = renderer_.render_forward_backward(N);
            capture_best(proposed_mse);
            if (proposed_mse > previous_mse) {
                params_ = std::move(before_relocation);
                renderer_.upload_params_gpu(params_.to_shape_params().data(), N);
                VIN_INFO("Rejected relocation: MSE %.3f -> %.3f", previous_mse, proposed_mse);
            }
            // Frozen layers took no Adam steps during the local phase. Restart
            // bias correction along with their moments before joint refinement.
            renderer_.reset_adam_state_all_gpu(N);
            adam_step_ = 0;
        } else {
            // Spend the full requested budget even when nothing needs recycling.
            // The last cycle always uses these steps for joint fine tuning.
            for (int li = 0; li < local_iters && !should_stop(); ++li) global_step();
        }
        VIN_INFO("Cycle %d: relocated=%zu, iterations=%d/%d", cycle, indices.size(),
                 completed_iters, config_.total_iters);
    }

    // Include the final update (or initial state when stopped immediately).
    capture_best(renderer_.render_forward_backward(N));
    VIN_CUDA(cudaEventSynchronize(snap_event_));
    for (int i = 0; i < N; ++i) {
        params_.cx[i] = snap_pinned_[i];
        params_.cy[i] = snap_pinned_[N + i];
        params_.rx[i] = snap_pinned_[2 * N + i];
        params_.ry[i] = snap_pinned_[3 * N + i];
        params_.angle[i] = snap_pinned_[4 * N + i];
        params_.color_r[i] = snap_pinned_[5 * N + 3 * i];
        params_.color_g[i] = snap_pinned_[5 * N + 3 * i + 1];
        params_.color_b[i] = snap_pinned_[5 * N + 3 * i + 2];
        params_.color_a[i] = snap_pinned_[8 * N + i] * 255.0f;
    }
    renderer_.upload_params_gpu(params_.to_shape_params().data(), N);
    renderer_.sync_stream();
    if (progress_cb_) progress_cb_(should_stop() ? static_cast<float>(completed_iters) / config_.total_iters : 1.0f,
                                  best_mse_, completed_iters);
    VIN_INFO("Restored best evaluated parameters: MSE=%.3f", best_mse_);
}

// ================================================================
// Cyclic Relocation
// ================================================================

void GradientOptimizer::find_relocation_candidates(
    const std::vector<float>& visibility,
    std::vector<int>& candidates,
    int& n_useful, int& n_useless, int& n_unstable)
{
    int N = params_.N;
    int window = config_.reloc_stability_window;

    if (history_count_ < window) {
        n_unstable = N;
        return;
    }

    // Download gradient history from GPU
    std::vector<float> h_history(window * N);
    VIN_CUDA(cudaMemcpy(h_history.data(), d_grad_history_, window * N * sizeof(float),
               cudaMemcpyDeviceToHost));

    // Reconstruct chronological order from ring buffer
    std::vector<float> chronological(window * N);
    int idx = history_write_idx_;
    if (idx > 0) {
        // [idx..end] + [0..idx)
        memcpy(chronological.data(), h_history.data() + idx * N,
               (window - idx) * N * sizeof(float));
        memcpy(chronological.data() + (window - idx) * N, h_history.data(),
               idx * N * sizeof(float));
    } else {
        chronological = h_history;
    }

    // Mean gradient norm per shape
    std::vector<float> mean_gn(N, 0.0f);
    for (int w = 0; w < window; ++w) {
        for (int i = 0; i < N; ++i) {
            mean_gn[i] += chronological[w * N + i];
        }
    }
    for (int i = 0; i < N; ++i) mean_gn[i] /= window;

    // Classify shapes: use blend weight sum (actual per-shape visibility)
    // Low visibility = too small OR buried under other shapes
    std::vector<std::pair<float, int>> cand_list;
    n_useful = n_useless = n_unstable = 0;

    for (int i = 0; i < N; ++i) {
        bool is_stable = mean_gn[i] < config_.reloc_stability_threshold;
        if (!is_stable) {
            n_unstable++;
            continue;
        }
        float vis = (i < static_cast<int>(visibility.size())) ? visibility[i] : 0.0f;
        bool is_useless = vis < 2.0f;
        if (is_useless) {
            n_useless++;
            cand_list.push_back({vis, i});
        } else {
            n_useful++;
        }
    }

    // Sort by visibility (lowest first)
    std::sort(cand_list.begin(), cand_list.end());
    candidates.clear();
    const int limit = std::max(1, static_cast<int>(N * config_.reloc_max_fraction));
    for (auto& [_, i] : cand_list) {
        if (static_cast<int>(candidates.size()) >= limit) break;
        candidates.push_back(i);
    }
}

std::vector<int> GradientOptimizer::relocate_shapes(OptParams& params,
                                                    const std::vector<int>& indices) {
    int n = static_cast<int>(indices.size());
    if (n == 0) return {};

    // Download pre-rendered d_output from GPU (avoids re-rendering)
    int total_pixels = H_opt_ * W_opt_;
    std::vector<float> rendered(total_pixels * 3);
    renderer_.download_output(rendered.data());
    std::vector<float> rendered_alpha(total_pixels), radius_scale(total_pixels);
    renderer_.download_alpha(rendered_alpha.data());

    // Match the weighted RGB objective, so relocation keeps prioritizing features.
    cv::Mat error_map = cv::Mat::zeros(H_opt_, W_opt_, CV_32FC1);
    for (int y = 0; y < H_opt_; ++y) {
        for (int x = 0; x < W_opt_; ++x) {
            int idx = y * W_opt_ + x;
            radius_scale[idx] = saliency_radius_scale(saliency_small_.at<float>(y, x));
            if (!mask_small_.at<uint8_t>(y, x)) continue;
            auto& tc = loss_target_small_.at<cv::Vec3f>(y, x);
            float dr = rendered[idx * 3 + 0] - tc[0];
            float dg = rendered[idx * 3 + 1] - tc[1];
            float db = rendered[idx * 3 + 2] - tc[2];
            // Missing coverage matters even where RGB matches the background.
            const float missing_alpha = std::max(0.0f,
                target_alpha_small_.at<float>(y, x) - rendered_alpha[idx]);
            error_map.at<float>(y, x) = (dr * dr + dg * dg + db * db)
                * rgb_loss_weights_.at<float>(y, x) + missing_alpha * missing_alpha;
        }
    }

    const int long_edge = std::max(W_opt_, H_opt_);
    const float size_lo = std::max(1.0f, long_edge * (config_.reloc_radius_lo / 100.0f));
    const float size_hi = std::max(size_lo, long_edge * (config_.reloc_radius_hi / 100.0f));
    const auto seeds = select_residual_seeds(error_map.ptr<float>(), W_opt_, H_opt_, n,
                                            size_lo, size_hi, radius_scale.data());
    n = static_cast<int>(seeds.size());
    if (n == 0) return {};

    std::vector<ShapeParams> replacements;
    replacements.reserve(n);
    for (const auto& seed : seeds) {
        ShapeParams shape{};
        shape.cx = static_cast<float>(seed.x); shape.cy = static_cast<float>(seed.y);
        shape.rx = seed.rx; shape.ry = seed.ry; shape.angle = seed.angle;
        auto [r, g, b] = get_local_color(shape.cx, shape.cy);
        shape.color_r = static_cast<float>(r); shape.color_g = static_cast<float>(g);
        shape.color_b = static_cast<float>(b);
        shape.opacity = target_alpha_small_.at<float>(seed.y, seed.x) / 255.0f;
        replacements.push_back(shape);
    }

    // Compact params_: remove relocated shapes, shift others forward
    int N = params.N;
    std::vector<bool> is_reloc(N, false);
    for (int j = 0; j < n; ++j) is_reloc[indices[j]] = true;

    auto compact = [&](auto& vec) {
        int w = 0;
        for (int i = 0; i < N; ++i) {
            if (!is_reloc[i]) {
                if (w != i) vec[w] = vec[i];
                ++w;
            }
        }
    };

    compact(params.cx);  compact(params.cy);
    compact(params.rx);  compact(params.ry);
    compact(params.angle);
    compact(params.color_r); compact(params.color_g);
    compact(params.color_b); compact(params.color_a);

    // Append relocated shapes at the end of params_ (new top layer)
    int base = N - n;
    std::vector<int> new_indices;
    for (int j = 0; j < n; ++j) {
        int dst = base + j;
        new_indices.push_back(dst);
        params.set_shape(dst, replacements[j]);
    }

    reloc_total_ += n;
    VIN_INFO("Relocated %d shapes to top layer (total: %d)", n, reloc_total_);
    return new_indices;
}

// ================================================================
// Parameter conversion
// ================================================================

std::vector<Shape> GradientOptimizer::params_to_shapes() {
    // Download params from GPU before converting
    auto sp = params_.to_shape_params();
    renderer_.download_params_gpu(sp.data(), params_.N);
    for (int i = 0; i < params_.N; ++i) params_.set_shape(i, sp[i]);

    std::vector<Shape> shapes;
    float inv_scale = 1.0f / scale_;

    for (int i = 0; i < params_.N; ++i) {
        int a_val = static_cast<int>(std::round(std::clamp(params_.color_a[i], 0.0f, 255.0f)));
        float opacity = a_val / 255.0f;
        int r = static_cast<int>(std::round(std::clamp(params_.color_r[i], 0.0f, 255.0f)));
        int g = static_cast<int>(std::round(std::clamp(params_.color_g[i], 0.0f, 255.0f)));
        int b = static_cast<int>(std::round(std::clamp(params_.color_b[i], 0.0f, 255.0f)));

        shapes.push_back(Shape::make_soft_ellipse_228(
            params_.cx[i] * inv_scale, params_.cy[i] * inv_scale,
            params_.rx[i] * inv_scale, params_.ry[i] * inv_scale,
            fmod(params_.angle[i], 360.0f),
            Color(r, g, b, 255), opacity));
    }

    return shapes;
}

cv::Mat GradientOptimizer::get_canvas() const {
    std::lock_guard<std::mutex> lock(viz_mutex_);
    int write_idx = viz_write_idx_.load();
    int read_idx = 1 - write_idx;
    return canvas_rgb_[read_idx].clone();
}

} // namespace vinylizer
