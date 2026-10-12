#include "inject/inject.h"
#include "common/logger.h"
#include "common/path_utils.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#endif

#include <fstream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <cmath>

namespace vinylizer {

// ================================================================
// Constants (matching Python inject.py)
// ================================================================

static constexpr uint64_t LIVERY_COUNT_OFFSET = 0x5A;
static constexpr uint64_t LAYER_TABLE_OFFSET = 0x78;
static constexpr uint64_t LAYER_POSITION_OFFSET = 0x18;
static constexpr uint64_t LAYER_SCALE_OFFSET = 0x28;
static constexpr uint64_t LAYER_ROTATION_OFFSET = 0x50;
static constexpr uint64_t LAYER_COLOR_OFFSET = 0x74;
static constexpr uint64_t LAYER_MASK_OFFSET = 0x78;
static constexpr uint64_t LAYER_SHAPE_ID_OFFSET = 0x7A;
static constexpr uint64_t LAYER_BLOB_SIZE = 0x80;

// shape_id mapping: JSON type -> FH6 memory shape_id
static const std::unordered_map<int, int> SHAPE_ID_MAP = {
    {1, 101}, {16, 102}, {228, 228}
};

// Scale divisor
static const std::unordered_map<int, int> SCALE_DIVISOR = {
    {1, 127}, {16, 63}, {228, 63}
};

static const DWORD ACCESS_MASK = PROCESS_VM_READ | PROCESS_VM_WRITE |
                                   PROCESS_VM_OPERATION | PROCESS_QUERY_INFORMATION;

// ================================================================
// Win32 memory helpers
// ================================================================

static bool read_mem(HANDLE h, uint64_t addr, void* buf, size_t size, size_t* nread = nullptr) {
    SIZE_T nr = 0;
    BOOL ok = ReadProcessMemory(h, reinterpret_cast<LPCVOID>(addr), buf, size, &nr);
    if (nread) *nread = nr;
    return ok && nr == size;
}

static bool write_mem(HANDLE h, uint64_t addr, const void* buf, size_t size) {
    SIZE_T nw = 0;
    return WriteProcessMemory(h, reinterpret_cast<LPVOID>(addr), buf, size, &nw) && nw == size;
}

static uint64_t read_u64(HANDLE h, uint64_t addr) {
    uint64_t v = 0;
    read_mem(h, addr, &v, 8);
    return v;
}

static bool is_user_pointer(uint64_t v) {
    return v >= 0x10000 && v <= 0x7FFFFFFFFFFF;
}

static bool is_private_writable(HANDLE h, uint64_t addr) {
    if (!is_user_pointer(addr)) return false;
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQueryEx(h, reinterpret_cast<LPCVOID>(addr), &mbi, sizeof(mbi)) == 0)
        return false;
    return mbi.State == MEM_COMMIT
        && mbi.Type == MEM_PRIVATE
        && !(mbi.Protect & PAGE_GUARD)
        && !(mbi.Protect & PAGE_NOACCESS)
        && (mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY |
                           PAGE_EXECUTE_READWRITE |
                           PAGE_EXECUTE_WRITECOPY));
}

static bool is_valid_layer_ptr(HANDLE h, uint64_t ptr) {
    if (!is_user_pointer(ptr)) return false;
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQueryEx(h, reinterpret_cast<LPCVOID>(ptr), &mbi, sizeof(mbi)) == 0)
        return false;
    if (!(mbi.State == MEM_COMMIT && !(mbi.Protect & PAGE_GUARD)
          && !(mbi.Protect & PAGE_NOACCESS)
          && (mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY |
                             PAGE_EXECUTE_READWRITE |
                             PAGE_EXECUTE_WRITECOPY))))
        return false;
    uint64_t region_end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
    return ptr + LAYER_BLOB_SIZE <= region_end;
}

// ================================================================
// Find FH6 process
// ================================================================

std::pair<int, std::string> find_fh6_process() {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return {0, ""};

    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof(pe);

    if (Process32FirstW(snap, &pe)) {
        do {
            std::wstring name(pe.szExeFile);
            std::string name_str(name.begin(), name.end());
            std::transform(name_str.begin(), name_str.end(), name_str.begin(), ::tolower);
            if (name_str == "forzahorizon6.exe" ||
                name_str == "forzahorizon6-win64-shipping.exe") {
                CloseHandle(snap);
                return {static_cast<int>(pe.th32ProcessID), name_str};
            }
        } while (Process32NextW(snap, &pe));
    }

    CloseHandle(snap);
    return {0, ""};
}

// ================================================================
// Validate JSON
// ================================================================

std::vector<nlohmann::json> validate_json(const std::string& path) {
    std::ifstream file = open_ifstream(path);
    if (!file.is_open()) {
        VIN_ERROR("Cannot open JSON file: %s", path.c_str());
        return {};
    }

    nlohmann::json data;
    try {
        data = nlohmann::json::parse(file);
    } catch (const nlohmann::json::exception& e) {
        VIN_ERROR("JSON parse error: %s", e.what());
        return {};
    }

    if (!data.is_object() || !data.contains("shapes") || !data["shapes"].is_array()) {
        VIN_ERROR("JSON must contain 'shapes' array");
        return {};
    }

    if (data["shapes"].empty() || data["shapes"].size() > MAX_LAYER_COUNT) {
        VIN_ERROR("Shape count must be in [1, %d]", MAX_LAYER_COUNT);
        return {};
    }
    std::vector<nlohmann::json> valid_shapes;
    for (const auto& shape : data["shapes"]) {
        std::string error;
        if (!validate_shape(shape, error)) {
            VIN_ERROR("Invalid shape[%zu]: %s", valid_shapes.size(), error.c_str());
            return {};
        }
        valid_shapes.push_back(shape);
    }

    VIN_INFO("JSON validated: %d shapes", static_cast<int>(valid_shapes.size()));
    return valid_shapes;
}

// ================================================================
// Strict layer check (replaces old score_layer)
// ================================================================

static bool strict_check_layer(HANDLE h, uint64_t ptr) {
    // 1. shape_id must be 101 (rectangle) or 102 (ellipse)
    uint8_t shape_id;
    if (!read_mem(h, ptr + LAYER_SHAPE_ID_OFFSET, &shape_id, 1)) return false;
    if (shape_id != 101 && shape_id != 102) return false;

    // 2. mask must be 0
    uint8_t mask;
    if (!read_mem(h, ptr + LAYER_MASK_OFFSET, &mask, 1)) return false;
    if (mask != 0) return false;

    // 3. color must be readable (uint8 naturally 0-255)
    uint8_t color[4];
    if (!read_mem(h, ptr + LAYER_COLOR_OFFSET, color, 4)) return false;

    // 4. position must be readable
    float pos[2];
    if (!read_mem(h, ptr + LAYER_POSITION_OFFSET, pos, 8)) return false;

    // 5. scale must be readable
    float scale[2];
    if (!read_mem(h, ptr + LAYER_SCALE_OFFSET, scale, 8)) return false;

    float rotation;
    if (!read_mem(h, ptr + LAYER_ROTATION_OFFSET, &rotation, 4)) return false;
    return std::isfinite(pos[0]) && std::isfinite(pos[1])
        && std::isfinite(scale[0]) && std::isfinite(scale[1])
        && scale[0] >= 0 && scale[1] >= 0 && std::isfinite(rotation);
}

// ================================================================
// Locate layer table
// ================================================================

uint64_t locate_layer_table(int pid, int layer_count, uint64_t* group_out) {
    if (group_out) *group_out = 0;
    if (layer_count <= 0 || layer_count > MAX_LAYER_COUNT) return 0;
    HANDLE h = OpenProcess(ACCESS_MASK, FALSE, pid);
    if (!h) {
        VIN_ERROR("Cannot open process %d", pid);
        return 0;
    }

    uint16_t pattern = static_cast<uint16_t>(layer_count);
    uint64_t address = 0x10000;
    MEMORY_BASIC_INFORMATION mbi{};

    while (address < 0x7FFFFFFFFFFF) {
        if (VirtualQueryEx(h, reinterpret_cast<LPCVOID>(address), &mbi, sizeof(mbi)) == 0) {
            address += 0x10000;
            continue;
        }

        uint64_t base = reinterpret_cast<uint64_t>(mbi.BaseAddress);
        size_t region_size = mbi.RegionSize;

        if (mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE
            && !(mbi.Protect & PAGE_GUARD) && !(mbi.Protect & PAGE_NOACCESS)
            && (mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                               PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                               PAGE_EXECUTE_WRITECOPY))) {

            size_t read_size = std::min(region_size, static_cast<size_t>(64 * 1024 * 1024));
            std::vector<uint8_t> mem(read_size);
            SIZE_T nr = 0;
            if (ReadProcessMemory(h, reinterpret_cast<LPCVOID>(base), mem.data(), read_size, &nr) && nr >= 2) {
                mem.resize(nr);
                for (size_t i = 0; i + 1 < mem.size(); ++i) {
                    uint16_t value;
                    std::memcpy(&value, &mem[i], sizeof(value));
                    if (value != pattern) continue;

                    uint64_t count_address = base + i;
                    uint64_t group_address = count_address - LIVERY_COUNT_OFFSET;
                    if (group_address < 0x10000) continue;

                    uint64_t table_address = read_u64(h, group_address + LAYER_TABLE_OFFSET);
                    if (!is_user_pointer(table_address)) continue;
                    if (!is_private_writable(h, table_address)) continue;

                    if (validate_layer_table_h(h, table_address, layer_count)) {
                        if (group_out) *group_out = group_address;
                        CloseHandle(h);
                        VIN_INFO("Found layer table at 0x%llX (group=0x%llX, all %d layers passed strict check)",
                                 table_address, group_address, layer_count);
                        return table_address;
                    }
                }
            }
        }

        uint64_t next = base + region_size;
        if (next <= address) break;
        address = next;
    }

    CloseHandle(h);
    VIN_ERROR("Failed to locate layer table");
    return 0;
}

// ================================================================
// Count valid layers and classify by shape_id
// ================================================================

bool layer_table_matches_h(HANDLE h, uint64_t group_address, uint64_t table_address, int layer_count) {
    uint16_t count = 0;
    uint64_t table = 0;
    return is_user_pointer(group_address) && layer_count > 0 && layer_count <= MAX_LAYER_COUNT
        && read_mem(h, group_address + LIVERY_COUNT_OFFSET, &count, sizeof(count))
        && read_mem(h, group_address + LAYER_TABLE_OFFSET, &table, sizeof(table))
        && count == layer_count && table == table_address;
}

bool validate_layer_table_h(HANDLE h, uint64_t table_address, int layer_count) {
    if (layer_count <= 0 || layer_count > MAX_LAYER_COUNT || !is_private_writable(h, table_address)) return false;
    std::unordered_set<uint64_t> pointers;
    for (int i = 0; i < layer_count; ++i) {
        uint64_t ptr = read_u64(h, table_address + static_cast<uint64_t>(i) * 8);
        if (!pointers.insert(ptr).second || !is_valid_layer_ptr(h, ptr) || !strict_check_layer(h, ptr)) return false;
    }
    return true;
}

LayerStats count_valid_layers(int pid, uint64_t table_address, int layer_count) {
    LayerStats stats;
    HANDLE h = OpenProcess(ACCESS_MASK, FALSE, pid);
    if (!h) return stats;

    for (int i = 0; i < layer_count; ++i) {
        uint64_t layer_ptr = read_u64(h, table_address + i * 8);
        if (!is_valid_layer_ptr(h, layer_ptr) || !strict_check_layer(h, layer_ptr)) continue;

        uint8_t shape_id;
        if (!read_mem(h, layer_ptr + LAYER_SHAPE_ID_OFFSET, &shape_id, 1)) continue;
        if (shape_id != 101 && shape_id != 102) continue;

        uint8_t mask;
        if (!read_mem(h, layer_ptr + LAYER_MASK_OFFSET, &mask, 1) || mask != 0) continue;

        stats.total_valid++;
        if (shape_id == 101) stats.count_101++;
        if (shape_id == 102) stats.count_102++;
    }

    CloseHandle(h);
    return stats;
}

// ================================================================
// Sample layers for user confirmation
// ================================================================

std::vector<LayerInfo> sample_layers(int pid, uint64_t table_address, int layer_count, int n) {
    std::vector<LayerInfo> result;
    HANDLE h = OpenProcess(ACCESS_MASK, FALSE, pid);
    if (!h) return result;

    // Generate random indices
    std::vector<int> indices;
    if (layer_count <= n) {
        for (int i = 0; i < layer_count; ++i) indices.push_back(i);
    } else {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        std::unordered_set<int> chosen;
        while (static_cast<int>(chosen.size()) < n) {
            int idx = std::rand() % layer_count;
            chosen.insert(idx);
        }
        indices.assign(chosen.begin(), chosen.end());
        std::sort(indices.begin(), indices.end());
    }

    for (int idx : indices) {
        uint64_t layer_ptr = read_u64(h, table_address + idx * 8);
        if (!is_valid_layer_ptr(h, layer_ptr)) continue;

        LayerInfo info;
        info.index = idx;
        info.ptr = layer_ptr;

        float pos[2], scale[2];
        if (!read_mem(h, layer_ptr + LAYER_POSITION_OFFSET, pos, 8)
            || !read_mem(h, layer_ptr + LAYER_SCALE_OFFSET, scale, 8)
            || !read_mem(h, layer_ptr + LAYER_ROTATION_OFFSET, &info.rotation, 4)
            || !read_mem(h, layer_ptr + LAYER_COLOR_OFFSET, info.color, 4)
            || !read_mem(h, layer_ptr + LAYER_SHAPE_ID_OFFSET, &info.shape_id, 1)
            || !read_mem(h, layer_ptr + LAYER_MASK_OFFSET, &info.mask, 1)) continue;
        info.pos_x = pos[0]; info.pos_y = pos[1];
        info.scale_x = scale[0]; info.scale_y = scale[1];

        result.push_back(info);
    }

    CloseHandle(h);
    return result;
}

// ================================================================
// Read layer info
// ================================================================

LayerInfo read_layer_info(int pid, uint64_t layer_ptr) {
    LayerInfo info;
    info.ptr = layer_ptr;

    HANDLE h = OpenProcess(ACCESS_MASK, FALSE, pid);
    if (!h) return info;

    float pos[2], scale[2];
    if (read_mem(h, layer_ptr + LAYER_POSITION_OFFSET, pos, 8)) {
        info.pos_x = pos[0]; info.pos_y = pos[1];
    }
    if (read_mem(h, layer_ptr + LAYER_SCALE_OFFSET, scale, 8)) {
        info.scale_x = scale[0]; info.scale_y = scale[1];
    }
    read_mem(h, layer_ptr + LAYER_ROTATION_OFFSET, &info.rotation, 4);
    read_mem(h, layer_ptr + LAYER_COLOR_OFFSET, info.color, 4);
    read_mem(h, layer_ptr + LAYER_SHAPE_ID_OFFSET, &info.shape_id, 1);
    read_mem(h, layer_ptr + LAYER_MASK_OFFSET, &info.mask, 1);

    CloseHandle(h);
    return info;
}

// ================================================================
// Write layer
// ================================================================

bool write_layer_h(HANDLE h, uint64_t table_address, int index, const nlohmann::json& shape_data) {
    std::string error;
    if (index < 0 || index >= MAX_LAYER_COUNT || !validate_shape(shape_data, error)) return false;
    uint64_t layer_ptr = read_u64(h, table_address + index * 8);
    if (!is_valid_layer_ptr(h, layer_ptr) || !strict_check_layer(h, layer_ptr)) {
        VIN_ERROR("Layer[%d] invalid pointer: 0x%llX", index, layer_ptr);
        return false;
    }

    int shape_type = shape_data["type"];
    auto& data = shape_data["data"];
    auto& color = shape_data["color"];

    float x = data[0], y = data[1], w = data[2], h_val = data[3];
    float rot_deg = data.size() >= 5 ? float(data[4]) : 0.0f;

    // Prepare write data
    float pos_data[2] = {x, -y};
    int divisor = SCALE_DIVISOR.count(shape_type) ? SCALE_DIVISOR.at(shape_type) : 63;
    float scale_data[2] = {w / divisor, h_val / divisor};
    float rot_data = std::fmod(360.0f - std::fmod(rot_deg, 360.0f), 360.0f);
    uint8_t color_data[4] = {
        static_cast<uint8_t>(color[0]), static_cast<uint8_t>(color[1]),
        static_cast<uint8_t>(color[2]), static_cast<uint8_t>(color[3])
    };
    uint8_t shape_id = static_cast<uint8_t>(SHAPE_ID_MAP.count(shape_type) ? SHAPE_ID_MAP.at(shape_type) : 102);
    uint8_t mask_data = 0;

    // Read old values for rollback
    float old_pos[2], old_scale[2], old_rot;
    uint8_t old_color[4], old_shape_id, old_mask;
    if (!read_mem(h, layer_ptr + LAYER_POSITION_OFFSET, old_pos, 8)
        || !read_mem(h, layer_ptr + LAYER_SCALE_OFFSET, old_scale, 8)
        || !read_mem(h, layer_ptr + LAYER_ROTATION_OFFSET, &old_rot, 4)
        || !read_mem(h, layer_ptr + LAYER_COLOR_OFFSET, old_color, 4)
        || !read_mem(h, layer_ptr + LAYER_SHAPE_ID_OFFSET, &old_shape_id, 1)
        || !read_mem(h, layer_ptr + LAYER_MASK_OFFSET, &old_mask, 1)) {
        VIN_ERROR("Layer[%d] backup read failed; nothing written", index);
        return false;
    }

    // Write new values with rollback on failure
    struct Write { uint64_t offset; const void* data; size_t size; const void* old_data; };
    Write writes[] = {
        {LAYER_POSITION_OFFSET, pos_data, 8, old_pos},
        {LAYER_SCALE_OFFSET, scale_data, 8, old_scale},
        {LAYER_ROTATION_OFFSET, &rot_data, 4, &old_rot},
        {LAYER_COLOR_OFFSET, color_data, 4, old_color},
        {LAYER_SHAPE_ID_OFFSET, &shape_id, 1, &old_shape_id},
        {LAYER_MASK_OFFSET, &mask_data, 1, &old_mask},
    };

    int completed = 0;
    for (auto& w : writes) {
        if (!write_mem(h, layer_ptr + w.offset, w.data, w.size)) {
            // Rollback
            for (int j = completed; j >= 0; --j) {
                if (!write_mem(h, layer_ptr + writes[j].offset, writes[j].old_data, writes[j].size))
                    VIN_ERROR("Layer[%d] rollback failed at offset 0x%llX", index, writes[j].offset);
            }
            VIN_ERROR("Layer[%d] write failed at offset 0x%llX", index, w.offset);
            return false;
        }
        completed++;
    }

    return true;
}

bool write_clear_layer_h(HANDLE h, uint64_t table_address, int index) {
    nlohmann::json clear_shape;
    clear_shape["type"] = 1;
    clear_shape["data"] = {0, 0, 0, 0, 0};
    clear_shape["color"] = {0, 0, 0, 0};
    return write_layer_h(h, table_address, index, clear_shape);
}

bool write_layer(int pid, uint64_t table_address, int index, const nlohmann::json& shape_data) {
    HANDLE h = OpenProcess(ACCESS_MASK, FALSE, pid);
    if (!h) {
        VIN_ERROR("Cannot open process %d for write_layer", pid);
        return false;
    }
    bool ok = write_layer_h(h, table_address, index, shape_data);
    CloseHandle(h);
    return ok;
}

bool write_clear_layer(int pid, uint64_t table_address, int index) {
    HANDLE h = OpenProcess(ACCESS_MASK, FALSE, pid);
    if (!h) {
        VIN_ERROR("Cannot open process %d for write_clear_layer", pid);
        return false;
    }
    bool ok = write_clear_layer_h(h, table_address, index);
    CloseHandle(h);
    return ok;
}

// ================================================================
// Main inject function
// ================================================================

bool inject_shapes(const InjectConfig& config) {
    // Validate JSON
    auto shapes = validate_json(config.json_path);
    if (shapes.empty()) return false;

    int layer_count = config.layer_count;
    if (!valid_layer_count(layer_count, shapes.size())) {
        VIN_ERROR("Placeholder count must exactly match JSON shape count (1..65535)");
        return false;
    }

    // Find FH6 process
    int pid = config.process_id;
    std::string proc_name;
    if (pid == 0) {
        auto [found_pid, found_name] = find_fh6_process();
        pid = found_pid;
        proc_name = found_name;
    }
    if (pid == 0) {
        VIN_ERROR("FH6 process not found");
        return false;
    }
    VIN_INFO("Found FH6: PID=%d", pid);

    // Locate layer table
    uint64_t group_address = 0;
    uint64_t table_address = locate_layer_table(pid, layer_count, &group_address);
    if (table_address == 0) return false;

    auto stats = count_valid_layers(pid, table_address, layer_count);
    if (stats.total_valid != layer_count) return false;
    VIN_INFO("Layers: %d rectangles, %d ellipses", stats.count_101, stats.count_102);
    auto samples = sample_layers(pid, table_address, layer_count, 6);
    if (samples.size() != static_cast<size_t>(std::min(layer_count, 6))) return false;
    for (const auto& s : samples) {
        VIN_INFO("Layer[%d] position=(%.2f,%.2f) scale=(%.4f,%.4f) angle=%.2f RGBA=(%u,%u,%u,%u) shape=%u mask=%u",
            s.index, s.pos_x, s.pos_y, s.scale_x, s.scale_y, s.rotation,
            s.color[0], s.color[1], s.color[2], s.color[3], s.shape_id, s.mask);
    }
    std::cout << "Verify these placeholders in FH6. Enter Y to import: " << std::flush;
    std::string confirmation;
    if (!std::getline(std::cin, confirmation) || (confirmation != "Y" && confirmation != "y")) return false;

    // Write shapes
    HANDLE h = OpenProcess(ACCESS_MASK, FALSE, pid);
    if (!h) {
        VIN_ERROR("Cannot open process %d for injection", pid);
        return false;
    }

    if (!layer_table_matches_h(h, group_address, table_address, layer_count)
        || !validate_layer_table_h(h, table_address, layer_count)) {
        VIN_ERROR("Layer table changed or failed validation; import cancelled");
        CloseHandle(h);
        return false;
    }
    int success = 0, fail = 0;

    for (int i = 0; i < layer_count; ++i) {
        bool ok = layer_table_matches_h(h, group_address, table_address, layer_count)
            && write_layer_h(h, table_address, i, shapes[i]);
        if (ok) success++; else fail++;

        if (fail > 0) {
            VIN_ERROR("Write failed; stopping after %d completed layers", success);
            break;
        }
    }

    CloseHandle(h);

    VIN_INFO("Inject complete: %d success, %d fail", success, fail);
    return fail == 0;
}

} // namespace vinylizer
