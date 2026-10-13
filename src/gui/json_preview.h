#pragma once

#include <QImage>
#include <string>

namespace vinylizer {

// Render the exported canvas, or a padded shape bounding box for legacy files
// without dimensions. The longest preview edge is capped at 2048 pixels.
// Invalid input or rendering failures return a null image.
QImage render_json_preview_file(const std::string& json_path);

} // namespace vinylizer
