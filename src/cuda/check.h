#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vinylizer {
inline void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
} // namespace vinylizer
#define VIN_CUDA(call) ::vinylizer::check_cuda((call), #call)
