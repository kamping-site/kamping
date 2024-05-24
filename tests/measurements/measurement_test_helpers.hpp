// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <string>
#include <unordered_map>
#include <vector>

#include "kamping/measurements/aggregated_tree_node.hpp"

namespace testing {

template <typename T = double>
struct AggregatedDataSummary {
    bool   is_scalar{true};
    bool   are_entries_consistent{true}; // number of values and value category are the same for all entries.
    size_t num_entries{0u};              // number of entries per measurement, might be greater then one if for example
                                         // counter/timer.append() has been called multiple times.
    size_t num_values_per_entry{
        0u}; // number of values per entry, might be greater than one if global aggregation mode "Gather" is executed.
    std::optional<std::vector<std::vector<T>>> aggregated_data;

    bool operator==(AggregatedDataSummary<T> const& other) const {
        bool const result =
            std::tie(is_scalar, are_entries_consistent, num_entries, num_values_per_entry, aggregated_data)
            == std::tie(
                other.is_scalar,
                other.are_entries_consistent,
                other.num_entries,
                other.num_values_per_entry,
                other.aggregated_data
            );
        return result;
    }
    auto& set_num_entries(size_t num_entries_) {
        num_entries = num_entries_;
        return *this;
    }
    auto& set_num_values_per_entry(size_t num_values) {
        num_values_per_entry = num_values;
        return *this;
    }
    auto& set_is_scalar(bool is_scalar_) {
        is_scalar = is_scalar_;
        return *this;
    }

    auto& set_aggregated_data(std::vector<std::vector<T>> param_aggregated_data) {
        aggregated_data = param_aggregated_data;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, AggregatedDataSummary const& summary) {
        return out << "is_scalar: " << std::boolalpha << summary.is_scalar
                   << ", entries_consistent: " << summary.are_entries_consistent
                   << ", #entries: " << summary.num_entries << ", #values per entry: " << summary.num_values_per_entry;
    }
};
template <typename T>
struct VisitorReturningSizeAndCategory {
    auto operator()(T const&) const {
        size_t size = 1u;
        return std::make_pair(size, true);
    }
    auto operator()(std::vector<T> const& vec) const {
        return std::make_pair(vec.size(), false);
    }
};

template <typename T>
struct VisitorReturningAlwaysVector {
    auto operator()(T const& elem) const {
        return std::vector<T>{elem};
    }
    auto operator()(std::vector<T> const& vec) const {
        return vec;
    }
};
// Traverses the evaluation tree and returns a summary of the aggregated data that can be used to verify to some degree
// the executed timings
template <typename T = double>
struct ValidationPrinter {
    void print(kamping::measurements::AggregatedTreeNode<T> const& node, bool store_only_metadata = true) {
        key_stack.push_back(node.name());
        for (auto const& [operation, aggregated_data]: node.aggregated_data()) {
            AggregatedDataSummary<T> summary;
            summary.num_entries = aggregated_data.size();
            if (aggregated_data.empty()) {
                continue;
            }
            {
                auto visitor                 = VisitorReturningSizeAndCategory<T>{};
                auto [size, is_scalar]       = std::visit(visitor, aggregated_data.front());
                summary.num_values_per_entry = size;
                summary.is_scalar            = is_scalar;
            }
            // check consistency of the entries if there are multiple
            {
                auto visitor = VisitorReturningSizeAndCategory<T>{};
                summary.are_entries_consistent =
                    std::all_of(aggregated_data.begin(), aggregated_data.end(), [&](auto const& entry) {
                        auto [size, is_scalar] = std::visit(visitor, entry);
                        return size == summary.num_values_per_entry && is_scalar == summary.is_scalar;
                    });
            }
            if (!store_only_metadata) {
                // is deactivated by default as measurements for timer events are not reproducible
                auto visitor            = VisitorReturningAlwaysVector<T>{};
                summary.aggregated_data = std::vector<std::vector<T>>{};
                for (auto const& entry: aggregated_data) {
                    summary.aggregated_data->push_back(std::visit(visitor, entry));
                }
            }

            output[concatenate_key_stack() + ":" + get_string(operation)] = summary;
        }

        for (auto const& child: node.children()) {
            if (child) {
                print(*child, store_only_metadata);
            }
        }
        key_stack.pop_back();
    }

    std::unordered_map<std::string, AggregatedDataSummary<T>> output;

private:
    std::vector<std::string> key_stack;

    std::string concatenate_key_stack() const {
        std::string str;
        for (auto const& key: key_stack) {
            if (!str.empty()) {
                str.append(".");
            }
            str.append(key);
        }
        return str;
    }
};
} // namespace testing
