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

#include <iostream>
#include <ostream>
#include <variant>
#include <vector>

#include "kamping/measurements/timer.hpp"

namespace kamping::measurements {

/// @brief Puts quotation marks around a given string.
/// @param str String to be quoted.
/// @return Quoted string.
inline std::string quote_string(std::string const& str) {
    return "\"" + str + "\"";
}

namespace internal {
/// @brief Able to print either a single value or a vector of value to the given outstream.
/// @tparam Value type to print.
template <typename T>
struct ScalarOrVectorPrinter {
    /// @brief Constructs a printer printing to the given outstream.
    /// @param outstream Outstream to print on.
    ScalarOrVectorPrinter(std::ostream& outstream) : _outstream{outstream} {}

    /// @brief Outputs the content of the given vector to outstream.
    /// @param vec Vector whose elements are comma-separated printed to the outstream.
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

    /// @brief Outputs the given scalar to outstream.
    /// @param scalar Scalar to be printed.
    void operator()(T const& scalar) const {
        _outstream << std::fixed << scalar;
    }
    std::ostream& _outstream; ///< Outstream used for printing.
};
} // namespace internal

/// @brief Printer class that prints an evaluated TimerTree in Json format.
///
/// @tparam Duration Type to represent a duration.
template <typename Duration = double>
class SimpleJsonPrinter {
public:
    /// @brief Construct a printer that use std::cout as outstream.
    SimpleJsonPrinter() : _outstream{std::cout}, _internal_printer(_outstream) {}

    /// @brief Construct a printer printing to a given outstream.
    ///
    /// @param outstream Outstream on which the content is printed.
    SimpleJsonPrinter(std::ostream& outstream) : _outstream{outstream}, _internal_printer(_outstream) {}

    /// @brief Construct a printer printing to a given outstream and adding
    /// additional configuration info.
    ///
    /// @param outstream Outstream on which the content is printed.
    /// @param config_info Additional configuration info (vector of key-value pairs), which will be printed as a
    /// "config" - dict by the printer.
    SimpleJsonPrinter(std::ostream& outstream, std::vector<std::pair<std::string, std::string>> config_info)
        : _outstream{outstream},
          _internal_printer(_outstream),
          _config_info{config_info} {}

    /// @brief Prints an evaluated TimerTree in Json format to stdout.
    /// @param node Root node of the TimerTree to print.
    /// @param indentation Indentation to use for the node.
    void print(AggregatedTreeNode<Duration> const& node, std::size_t indentation = 0) {
        _outstream << std::string(indentation, ' ') << "{" << std::endl;
        _outstream << std::string(indentation + indentation_per_level, ' ') << quote_string("data") << ": {"
                   << std::endl;
        print_impl(node, indentation + indentation_per_level + indentation_per_level);
        _outstream << std::endl;
        _outstream << std::string(indentation + indentation_per_level, ' ') << "}," << std::endl;
        print_config(indentation + indentation_per_level);

        _outstream << std::string(indentation, ' ') << "}";
    }

private:
    std::ostream&                             _outstream; ///< Outstream to print on.
    std::size_t                               indentation_per_level = 2u;
    internal::ScalarOrVectorPrinter<Duration> _internal_printer; ///< Internal printer able to print either a scalar or
                                                                 ///< vector of durations.
    std::vector<std::pair<std::string, std::string>> _config_info;

    void print_config(std::size_t indentation) {
        _outstream << std::boolalpha;
        _outstream << std::string(indentation, ' ') << quote_string("config") << ": {";
        if (_config_info.empty()) {
            // close dict in same line if config is empty and return
            _outstream << "}" << std::endl;
            return;
        }
        // otherwise output config in new line
        _outstream << std::endl;
        bool is_first = true;
        for (auto const& [key, value]: _config_info) {
            if (!is_first) {
                _outstream << "," << std::endl;
            }
            is_first = false;
            _outstream << std::string(indentation + indentation_per_level, ' ') << quote_string(key) << ":"
                       << quote_string(value);
        }
        _outstream << std::endl << std::string(indentation, ' ') << "}" << std::endl;
    }

    /// @brief Prints an evaluated TimerTree in Json format to stdout.
    /// @param node Root node of the TimerTree to print.
    /// @param indentation Indentation to use for the node.
    void print_impl(AggregatedTreeNode<Duration> const& node, std::size_t indentation = 0) {
        auto name            = node.name();
        auto evaluation_data = node.aggregated_data();
        _outstream << std::string(indentation, ' ') << quote_string(name) << ": {" << std::endl;
        _outstream << std::string(indentation + indentation_per_level, ' ') << quote_string("statistics") << ": {"
                   << std::endl;
        if (!evaluation_data.empty()) {
            bool is_first_outer = true;
            for (auto const& [op, data]: evaluation_data) {
                if (!is_first_outer) {
                    _outstream << "," << std::endl;
                }
                is_first_outer = false;
                _outstream << std::string(indentation + 2 * indentation_per_level, ' ') << "\"" << get_string(op)
                           << "\""
                           << ": [";
                bool is_first = true;
                for (auto const& data_item: data) {
                    if (!is_first) {
                        _outstream << ", ";
                    }
                    is_first = false;
                    std::visit(_internal_printer, data_item);
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
            print_impl(*children, indentation + indentation_per_level);
        }
        if (!node.children().empty()) {
            _outstream << std::endl;
        }
        _outstream << std::string(indentation, ' ') << "}";
    }
};

/// @brief Printer class that prints an evaluated TimerTree in a flat format in which the timer hierarchy is collapsed
/// into a dot separated identifier per measurement.
///
/// For example:
/// \code
///   timer.start("algo")
///     timer.start("subroutine")
///       timer.start("subsubroutine")
///       timer.stop()
///     timer.stop()
///   timer.stop()
/// \endcode
/// will return an output conceptually similar to
/// \code
/// // algo=<duration data> algo.subroutine= <duration data> algo.subroutine.subsubroutine= <duration data> ...
/// \endcode
/// when printed with FlatPrinter.
///
class FlatPrinter {
public:
    /// @brief Construct a printer that use std::cout as outstream.
    FlatPrinter() : _outstream{std::cout} {}

    /// @brief Construct a printer printing to a given outstream.
    ///
    /// @param outstream Outstream on which the content is printed.
    FlatPrinter(std::ostream& outstream) : _outstream{outstream} {}
    /// @brief Prints an evaluated TimerTree in Json format to stdout.
    /// @tparam Duration Type to represent a duration.
    /// @param node Root node of the TimerTree to print.
    template <typename Duration>
    void print(AggregatedTreeNode<Duration> const& node) {
        _key_stack.push_back(node.name());
        internal::ScalarOrVectorPrinter<Duration> internal_printer{_outstream};
        for (auto const& [operation, aggregated_data]: node.aggregated_data()) {
            _outstream << " " << concatenate_key_stack() << ":" << get_string(operation) << "=[";
            bool is_first = true;
            for (auto const& data_item: aggregated_data) {
                if (!is_first) {
                    _outstream << ", ";
                }
                is_first = false;
                std::visit(internal_printer, data_item);
            }
            _outstream << "]";
        }

        for (auto const& child: node.children()) {
            if (child) {
                print(*child);
            }
        }
        _key_stack.pop_back();
    }

private:
    std::ostream&            _outstream; ///< Outstream to print on.
    std::vector<std::string> _key_stack; ///< Stack tracking the current path to the root.

    std::string concatenate_key_stack() const {
        std::string str;
        for (auto const& key: _key_stack) {
            if (!str.empty()) {
                str.append(".");
            }
            str.append(key);
        }
        return str;
    }
};

} // namespace kamping::measurements
