#include "gui/config_validation.h"

#include <iostream>
#include <stdexcept>

namespace {
using vinylizer::gui::parse_inject_layer_count;
using vinylizer::gui::parse_optimization_inputs;
using Fields = std::array<std::string, 6>;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Fields valid_fields() { return {"64", "80", "2", "32", "0.1", "0.01"}; }

void valid_inputs() {
    const auto parsed = parse_optimization_inputs(valid_fields());
    require(parsed.has_value(), "Valid optimization inputs rejected");
    require(parsed->canvas_w == 64 && parsed->canvas_h == 80
        && parsed->num_cycles == 2 && parsed->num_shapes == 32
        && parsed->lr_start == 0.1f && parsed->lr_end == 0.01f,
        "Parsed optimization values differ from the displayed inputs");

    const std::string canvas_fields[6] = {"64", "80", "2", "32", "0.1", "0.01"};
    require(parse_optimization_inputs(canvas_fields).has_value(),
            "RenderCanvas input array is not accepted");

    Fields limits = {std::to_string((std::numeric_limits<int>::max)()),
                     std::to_string((std::numeric_limits<int>::max)()),
                     std::to_string((std::numeric_limits<int>::max)() / 200),
                     "65535", "1", "1"};
    require(parse_optimization_inputs(limits).has_value(), "Valid upper bounds rejected");
    require(parse_optimization_inputs(Fields{"1", "1", "1", "1", "1", "1"}).has_value(),
            "Valid lower bounds rejected");
    require(parse_optimization_inputs(Fields{"64", "80", "2", "0032", ".1", "1e-2"}).has_value(),
            "Decimal or scientific learning rates rejected");
    require(parse_inject_layer_count("1") == 1 && parse_inject_layer_count("65535") == 65535,
            "Injection layer limits rejected");
}

void invalid_optimization_inputs() {
    for (size_t field = 0; field < 6; ++field) {
        for (const std::string bad : {"", " ", "\t", " 1", "1 ", "1\n", "0", "-1", "1x"}) {
            auto inputs = valid_fields();
            inputs[field] = bad;
            require(!parse_optimization_inputs(inputs),
                    "Accepted invalid field " + std::to_string(field) + ": [" + bad + "]");
        }
        auto embedded_nul = valid_fields();
        embedded_nul[field] = std::string("1\0junk", 6);
        require(!parse_optimization_inputs(embedded_nul), "Accepted text after an embedded NUL");
    }
    for (size_t field = 0; field < 4; ++field) {
        for (const std::string bad : {"32.5", "32.", ".5", "1e2", "+32", "99999999999999999999999"}) {
            auto inputs = valid_fields();
            inputs[field] = bad;
            require(!parse_optimization_inputs(inputs),
                    "Accepted invalid integer field " + std::to_string(field) + ": " + bad);
        }
    }
    auto inputs = valid_fields();
    inputs[2] = std::to_string((std::numeric_limits<int>::max)() / 200 + 1);
    require(!parse_optimization_inputs(inputs), "Cycle count overflows the iteration budget");
    inputs = valid_fields();
    inputs[3] = "65536";
    require(!parse_optimization_inputs(inputs), "Accepted more than 65535 shapes");

    for (size_t field = 4; field < 6; ++field) {
        for (const std::string bad : {"nan", "NaN", "inf", "Infinity", "-inf", "1e999",
                                      "1e-999", "-0", "0.0", "1.2.3", "0x1p2"}) {
            inputs = valid_fields();
            inputs[field] = bad;
            require(!parse_optimization_inputs(inputs), "Accepted invalid learning rate: " + bad);
        }
    }
    inputs = valid_fields();
    inputs[5] = "0.2";
    require(!parse_optimization_inputs(inputs), "Accepted an end learning rate above the start");

    auto current = parse_optimization_inputs(valid_fields());
    require(current.has_value(), "Initial configuration rejected");
    inputs = valid_fields();
    inputs[3].clear();
    current = parse_optimization_inputs(inputs);
    require(!current, "An invalid edit retained a previous optimization configuration");
}

void invalid_injection_inputs() {
    for (const std::string bad : {"", " ", "\t", "32 ", " 32", "0", "-1", "+32", "32.5",
                                  "32junk", "1e2", "65536", "2147483648",
                                  "99999999999999999999999", "NaN", "Inf"}) {
        require(!parse_inject_layer_count(bad), "Accepted invalid injection layer count: " + bad);
    }
    require(!parse_inject_layer_count(std::string_view("32\0junk", 7)),
            "Accepted injection text after an embedded NUL");
    auto current = parse_inject_layer_count("32");
    require(current == 32, "Initial injection layer count rejected");
    current = parse_inject_layer_count("");
    require(!current, "An empty edit retained the previous injection layer count");
}
} // namespace

int main() {
    try {
        valid_inputs();
        invalid_optimization_inputs();
        invalid_injection_inputs();
        std::cout << "GUI configuration validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
