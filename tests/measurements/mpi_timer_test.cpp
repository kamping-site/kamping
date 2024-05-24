// This file is part of KaMPIng.
//
// Copyright 2023-2024 The KaMPIng Authors
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

#include "kamping/measurements/timer.hpp"
#include "measurement_test_helpers.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::measurements;

TEST(TimerTest, basics) {
    auto const& comm = comm_world();
    Timer<>     timer;
    timer.start("measurement");
    timer.stop();
    auto              aggregated_timer_tree = timer.aggregate();
    ValidationPrinter printer;
    printer.print(aggregated_timer_tree.root());

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement:max",
             AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(1).set_is_scalar(true)}};
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
    printer.print(aggregated_timer_tree.root());

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement:max",
             AggregatedDataSummary{}.set_num_entries(2).set_num_values_per_entry(1).set_is_scalar(true)}};
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
    printer.print(aggregated_timer_tree.root());

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement:max",
             AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(1).set_is_scalar(true)}};
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
    printer.print(aggregated_timer_tree.root());

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement:max",
             AggregatedDataSummary{}.set_num_entries(2).set_num_values_per_entry(1).set_is_scalar(true)},
            {"root.measurement:min",
             AggregatedDataSummary{}.set_num_entries(2).set_num_values_per_entry(1).set_is_scalar(true)},
            {"root.measurement:gather",
             AggregatedDataSummary{}.set_num_entries(2).set_num_values_per_entry(comm.size()).set_is_scalar(false)}};

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
    printer.print(aggregated_timer_tree.root());

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement:max",
             AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(1).set_is_scalar(true)},
            {"root.measurement:min",
             AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(1).set_is_scalar(true)},
            {"root.measurement:gather",
             AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(comm.size()).set_is_scalar(false)}};
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
    printer.print(aggregated_timer_tree.root());

    if (comm.is_root()) {
        auto const expected_summary =
            AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(1).set_is_scalar(true);
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
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
    printer.print(aggregated_timer_tree.root());

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement1:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(repetitions).set_num_values_per_entry(1)},
            {"root.measurement1.measurement12:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values_per_entry(1)},
            {"root.measurement1.measurement12.measurement121:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values_per_entry(1)},
            {"root.measurement1.measurement11:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values_per_entry(1)},
            {"root.measurement1.measurement11:gather",
             AggregatedDataSummary{}.set_is_scalar(false).set_num_entries(1).set_num_values_per_entry(comm.size())}};
        EXPECT_EQ(printer.output, expected_output);
    };
}

TEST(TimerTest, print) {
    size_t const      repetitions           = 5u;
    auto              timer1                = setup_complex_scenario(repetitions);
    auto              timer2                = setup_complex_scenario(repetitions);
    auto              aggregated_timer_tree = timer1.aggregate();
    ValidationPrinter printer1;
    printer1.print(aggregated_timer_tree.root());
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
        printer.print(aggregated_timer_tree.root());

        if (split_comm.is_root()) {
            std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
                {"root.measurement:max",
                 AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(1).set_is_scalar(true)}};
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
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement:max",
             AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(1).set_is_scalar(true)}};
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
    printer.print(aggregated_timer_tree.root());

    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement:max",
             AggregatedDataSummary{}.set_num_entries(1).set_num_values_per_entry(1).set_is_scalar(true)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(TimerTest, enable_disable) {
    Communicator<> comm;
    Timer<>        timer;
    timer.disable();
    timer.start("measurement1");
    timer.enable();
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
    timer.disable();
    timer.stop_and_append();
    timer.enable();
    ValidationPrinter printer;
    timer.aggregate_and_print(printer);
    if (comm.is_root()) {
        std::unordered_map<std::string, AggregatedDataSummary<>> expected_output{
            {"root.measurement11:gather",
             AggregatedDataSummary{}.set_is_scalar(false).set_num_entries(1).set_num_values_per_entry(comm.size())},
            {"root.measurement12:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values_per_entry(1)},
            {"root.measurement12.measurement121:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values_per_entry(1)},
            {"root.measurement11:max",
             AggregatedDataSummary{}.set_is_scalar(true).set_num_entries(1).set_num_values_per_entry(1)},
        };
        EXPECT_EQ(printer.output, expected_output);
    };
}
