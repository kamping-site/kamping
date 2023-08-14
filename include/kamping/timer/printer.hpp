// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include "kamping/timer/timer.hpp"

namespace kamping::timer {

/// @brief Puts quotation marks around a given string.
/// @param str String to be quoted.
/// @return Quoted string.
inline std::string quote_string(std::string const& str) {
    return "\"" + str + "\"";
}

/// @brief Printer class that prints an evaluated TimerTree in Json format.
class SimpleJsonPrinter {
public:
    /// @brief Construct a printer that use std::cout as outstream.
    SimpleJsonPrinter() : _outstream{std::cout} {}

    /// @brief Construct a printer printing to a given outstream.
    ///
    /// @param outstream Outstream on which the content is printed.
    SimpleJsonPrinter(std::ostream& outstream) : _outstream{outstream} {}
    /// @brief Prints an evaluated TimerTree in Json format to stdout.
    /// @tparam Duration Type to represent a duration.
    /// @param node Root node of the TimerTree to print.
    /// @param indentation Indentation to use for the node.
    template <typename Duration>
    void print(EvaluationTreeNode<Duration> const& node, std::size_t indentation = 0) {
        const std::size_t indentation_per_level = 2;
        auto              name                  = node.name();
        auto              evaluation_data       = node.aggregated_data();
        _outstream << std::string(indentation, ' ') << quote_string(name) << ": {" << std::endl;

        InternalPrinter<Duration> internal_printer{_outstream};
        _outstream << std::string(indentation + indentation_per_level, ' ') << quote_string("statistics") << ": {"
                   << std::endl;
        if (!evaluation_data.empty()) {
            bool is_first_outer = true;
            for (auto const& [op, data]: evaluation_data) {
                if (!is_first_outer) {
                    _outstream << "," << std::endl;
                }
                is_first_outer = false;
                _outstream << std::string(indentation + 2 * indentation_per_level, ' ') << "\"" << op << "\""
                           << ": [";
                bool is_first = true;
                for (auto const& data_item: data) {
                    if (!is_first) {
                        _outstream << ", ";
                    }
                    is_first = false;
                    std::visit(internal_printer, data_item);
                }
                _outstream << "]";
            }
            _outstream << std::endl;
        }
        _outstream << std::string(indentation + indentation_per_level, ' ') << "}";
        if (!node.children().empty()) {
            _outstream << ",";
        }
        _outstream << std::endl;

        bool is_first = true;
        for (auto const& children: node.children()) {
            if (!is_first) {
                _outstream << "," << std::endl;
            }
            is_first = false;
            print(*children, indentation + indentation_per_level);
        }
        if (!node.children().empty()) {
            _outstream << std::endl;
        }
        _outstream << std::string(indentation, ' ') << "}";
    }

private:
    std::ostream& _outstream; ///< Outstream to print on.
    template <typename T>
    struct InternalPrinter {
        InternalPrinter(std::ostream& outstream) : _outstream{outstream} {}
        void operator()(std::vector<T> const& vec) const {
            _outstream << "[";
            bool is_first = true;
            for (auto const& elem: vec) {
                if (!is_first) {
                    _outstream << ", ";
                }
                is_first = false;
                _outstream << std::fixed << elem;
            }
            _outstream << "]";
        }
        void operator()(T const& scalar) const {
            _outstream << std::fixed << scalar;
        }
        std::ostream& _outstream;
    };
};

} // namespace kamping::timer
