#pragma once

#include "common/types.h"
#include <string>
#include <vector>

namespace vinylizer {

// Throws std::runtime_error if opening or completing the write fails.
void write_json(const std::string& path, const std::vector<Shape>& shapes,
                int canvas_w, int canvas_h);

} // namespace vinylizer
