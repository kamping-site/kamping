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

#include "gmock/gmock.h"
#include <algorithm>
#include <cstddef>
#include <stack>
#include <tuple>
#include <unordered_map>

#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/measurements/timer.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::measurements;

struct AggregatedDataSummary {
    bool   is_scalar{true};
    bool   are_entries_consistent{true}; // number of values and value category are the same for all entries.
    size_t num_entries{0u};
    size_t num_values_per_entry{0u};
    bool   operator==(AggregatedDataSummary const& other) const {
          bool const result =
              std::tie(is_scalar, are_entries_consistent, num_entries, num_values_per_entry)
              == std::tie(other.is_scalar, other.are_entries_consistent, other.num_entries, other.num_values_per_entry);
          return result;
    }
    auto& set_num_entries(size_t num_entries_) {
        num_entries = num_entries_;
        return *this;
    }
    auto& set_num_values(size_t num_values) {
        num_values_per_entry = num_values;
        return *this;
    }
    auto& set_is_scalar(bool is_scalar_) {
        is_scalar = is_scalar_;
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
// Traverses the evaluation tree and returns a smmary of the aggregated data that can be used to verify to some degree
// the executed timings
struct ValidationPrinter {
    void print(measurements::AggregatedTreeNode<double> const& node) {
        key_stack.push_back(node.name());
        for (auto const& [operation, aggregated_data]: node.aggregated_data()) {
            AggregatedDataSummary summary;
            summary.num_entries = aggregated_data.size();
            if (aggregated_data.empty()) {
                continue;
            }
            auto visitor = VisitorReturningSizeAndCategory<double>{};
            {
                auto [size, is_scalar]       = std::visit(visitor, aggregated_data.front());
                summary.num_values_per_entry = size;
                summary.is_scalar            = is_scalar;
            }
            // check consistency of the entries if there are multiple
            summary.are_entries_consistent =
                std::all_of(aggregated_data.begin(), aggregated_data.end(), [&](auto const& entry) {
                    auto [size, is_scalar] = std::visit(visitor, entry);
                    return size == summary.num_values_per_entry && is_scalar == summary.is_scalar;
                });

            output[concatenate_key_stack() + ":" + get_string(operation)] = summary;
        }

        for (auto const& child: node.children()) {
            if (child) {
                print(*child);
            }
        }
        key_stack.pop_back();
    }

    std::unordered_map<std::string, AggregatedDataSummary> output;

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

TEST(TimerTest, basics) {
    auto const& comm = comm_world();
    Timer<>     timer;
    timer.start("measurement");
    timer.stop();
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree);

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement:max", AggregatedDataSummary{}.set_num_entries(1).set_num_values(1).set_is_scalar(true)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(TimerTest, basics_append) {
    auto const& comm = comm_world();
    Timer<>     timer;
    timer.start("measurement");
    timer.stop();
    timer.start("measurement");
    timer.stop_and_append();
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree);

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement:max", AggregatedDataSummary{}.set_num_entries(2).set_num_values(1).set_is_scalar(true)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(TimerTest, basics_accumulate) {
    auto const& comm = comm_world();
    Timer<>     timer;
    timer.start("measurement");
    timer.stop();
    timer.start("measurement");
    timer.stop_and_add();
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree);

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement:max", AggregatedDataSummary{}.set_num_entries(1).set_num_values(1).set_is_scalar(true)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(TimerTest, stop_and_append_multiple_operations) {
    auto const& comm = comm_world();
    Timer<>     timer;
    timer.start("measurement");
    timer.stop();
    timer.start("measurement");
    timer.stop_and_append({GlobalAggregationMode::max, GlobalAggregationMode::min, GlobalAggregationMode::gather});
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree);

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement:max", AggregatedDataSummary{}.set_num_entries(2).set_num_values(1).set_is_scalar(true)},
            {"root.measurement:min", AggregatedDataSummary{}.set_num_entries(2).set_num_values(1).set_is_scalar(true)},
            {"root.measurement:gather",
             AggregatedDataSummary{}.set_num_entries(2).set_num_values(comm.size()).set_is_scalar(false)}};

        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(TimerTest, stop_and_add_multiple_operations) {
    auto const& comm = comm_world();
    Timer<>     timer;
    timer.start("measurement");
    timer.stop();
    timer.start("measurement");
    timer.stop_and_add({GlobalAggregationMode::max, GlobalAggregationMode::min, GlobalAggregationMode::gather});
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree);

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement:max", AggregatedDataSummary{}.set_num_entries(1).set_num_values(1).set_is_scalar(true)},
            {"root.measurement:min", AggregatedDataSummary{}.set_num_entries(1).set_num_values(1).set_is_scalar(true)},
            {"root.measurement:gather",
             AggregatedDataSummary{}.set_num_entries(1).set_num_values(comm.size()).set_is_scalar(false)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(TimerTest, stop_nested_scenario) {
    auto const& comm = comm_world();
    Timer<>     timer;
    timer.start("measurement1");
    {
        timer.start("measurement11");
        timer.stop();
        timer.start("measurement12");
        timer.stop();
    }
    timer.stop();
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree);

    if (comm.is_root()) {
        auto const expected_summary = AggregatedDataSummary{}.set_num_entries(1).set_num_values(1).set_is_scalar(true);
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement1:max", expected_summary},
            {"root.measurement1.measurement11:max", expected_summary},
            {"root.measurement1.measurement12:max", expected_summary}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

auto setup_complex_scenario(size_t repetitions) {
    Timer<> timer;
    for (size_t i = 0; i < repetitions; ++i) {
        timer.start("measurement1");
        {
            timer.start("measurement11");
            timer.stop({measurements::GlobalAggregationMode::gather, measurements::GlobalAggregationMode::max});
            timer.start("measurement12");
            {
                timer.synchronize_and_start("measurement121");
                timer.stop();
            }
            timer.stop();
            timer.start("measurement11");
            timer.stop();
        }
        timer.stop_and_append();
    }
    return timer;
}

TEST(TimerTest, stop_nested_complex_scenario) {
    auto const&       comm                  = comm_world();
    size_t            repetitions           = 5u;
    auto              timer                 = setup_complex_scenario(repetitions);
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree);

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement1:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(repetitions).set_num_values(1)},
            {"root.measurement1.measurement12:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values(1)},
            {"root.measurement1.measurement12.measurement121:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values(1)},
            {"root.measurement1.measurement11:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values(1)},
            {"root.measurement1.measurement11:gather",
             AggregatedDataSummary{}.set_is_scalar(false).set_num_entries(1).set_num_values(comm.size())}};
        EXPECT_EQ(printer.output, expected_output);
    };
}

TEST(TimerTest, print) {
    size_t const      repetitions           = 5u;
    auto              timer1                = setup_complex_scenario(repetitions);
    auto              timer2                = setup_complex_scenario(repetitions);
    auto              aggregated_timer_tree = timer1.aggregate();
    ValidationPrinter printer1;
    printer1.print(aggregated_timer_tree);
    ValidationPrinter printer2;
    timer2.aggregate_and_print(printer2);
    EXPECT_EQ(printer1.output, printer2.output);
}

TEST(TimerTest, synchronize_and_start_non_trivial_communicator) {
    auto const& comm       = comm_world();
    int const   color      = comm.rank() % 2;
    auto        split_comm = comm.split(color);
    Timer<>     timer(split_comm);
    // checks (among other things) that synchronize uses the subcommunicator for the barrier
    if (color == 0) {
        timer.synchronize_and_start("measurement");
    }
}

TEST(TimerTest, aggregate_non_trivial_communicator) {
    auto const& comm       = comm_world();
    int const   color      = comm.rank() % 2;
    auto        split_comm = comm.split(color);
    Timer<>     timer(split_comm);
    if (color == 0) {
        timer.synchronize_and_start("measurement");
        timer.stop();
        auto              aggregated_timer_tree = timer.aggregate();
        ValidationPrinter printer;
        printer.print(aggregated_timer_tree);

        if (split_comm.is_root()) {
            std::unordered_map<std::string, AggregatedDataSummary> expected_output{
                {"root.measurement:max",
                 AggregatedDataSummary{}.set_num_entries(1).set_num_values(1).set_is_scalar(true)}};
            EXPECT_EQ(printer.output, expected_output);
        }
    }
}

TEST(TimerTest, aggregate_and_print_non_trivial_communicator) {
    auto const& comm       = comm_world();
    int const   color      = comm.rank() % 2;
    auto        split_comm = comm.split(color);
    Timer<>     timer(split_comm);
    timer.synchronize_and_start("measurement");
    timer.stop();
    ValidationPrinter printer;
    timer.aggregate_and_print(printer);

    if (split_comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement:max", AggregatedDataSummary{}.set_num_entries(1).set_num_values(1).set_is_scalar(true)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(TimerTest, clear) {
    Communicator<> comm;
    size_t const   repetitions = 5u;
    auto           timer       = setup_complex_scenario(repetitions);
    timer.clear();
    ValidationPrinter printer;
    timer.aggregate_and_print(printer);
    if (comm.is_root()) {
        EXPECT_EQ(printer.output.size(), 0u);
    };
}

TEST(TimerTest, singleton) {
    Communicator<> comm;
    auto&          timer = kamping::measurements::timer();
    timer.clear();
    timer.start("measurement");
    timer.stop();
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree);

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary> expected_output{
            {"root.measurement:max", AggregatedDataSummary{}.set_num_entries(1).set_num_values(1).set_is_scalar(true)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}
