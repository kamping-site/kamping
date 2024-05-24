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

#include <unordered_map>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/measurements/counter.hpp"
#include "measurement_test_helpers.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::measurements;

TEST(CounterTest, basics) {
    auto const& comm = comm_world();
    Counter     counter;
    using DataType = decltype(counter)::DataType;
    counter.add("measurement", 42);
    auto                        aggregated_counter_tree = counter.aggregate();
    ValidationPrinter<DataType> printer;
    printer.print(aggregated_counter_tree.root(), false);
    if (comm.is_root()) {
        std::vector<std::vector<DataType>> const                         expected_data{{42 * comm.size_signed()}};
        std::unordered_map<std::string, AggregatedDataSummary<DataType>> expected_output{
            {"root.measurement:sum",
             AggregatedDataSummary<DataType>{}
                 .set_num_entries(1)
                 .set_num_values_per_entry(1)
                 .set_is_scalar(true)
                 .set_aggregated_data(expected_data)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(CounterTest, max_aggregation) {
    auto const& comm = comm_world();
    Counter     counter;
    using DataType = decltype(counter)::DataType;
    counter.add("measurement", comm.rank_signed() + 1, {GlobalAggregationMode::max});
    auto                        aggregated_counter_tree = counter.aggregate();
    ValidationPrinter<DataType> printer;
    printer.print(aggregated_counter_tree.root(), false);
    if (comm.is_root()) {
        std::vector<std::vector<DataType>> const                         expected_data{{comm.size_signed()}};
        std::unordered_map<std::string, AggregatedDataSummary<DataType>> expected_output{
            {"root.measurement:max",
             AggregatedDataSummary<DataType>{}
                 .set_num_entries(1)
                 .set_num_values_per_entry(1)
                 .set_is_scalar(true)
                 .set_aggregated_data(expected_data)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(CounterTest, min_aggregation) {
    auto const& comm = comm_world();
    Counter     counter;
    using DataType = decltype(counter)::DataType;
    counter.add("measurement", comm.rank_signed() + 1, {GlobalAggregationMode::min});
    auto                        aggregated_counter_tree = counter.aggregate();
    ValidationPrinter<DataType> printer;
    printer.print(aggregated_counter_tree.root(), false);
    if (comm.is_root()) {
        std::vector<std::vector<DataType>> const                         expected_data{{1}};
        std::unordered_map<std::string, AggregatedDataSummary<DataType>> expected_output{
            {"root.measurement:min",
             AggregatedDataSummary<DataType>{}
                 .set_num_entries(1)
                 .set_num_values_per_entry(1)
                 .set_is_scalar(true)
                 .set_aggregated_data(expected_data)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(CounterTest, sum_aggregation) {
    auto const& comm = comm_world();
    Counter     counter;
    using DataType = decltype(counter)::DataType;
    counter.add("measurement", comm.rank_signed() + 1, {GlobalAggregationMode::sum});
    auto                        aggregated_counter_tree = counter.aggregate();
    ValidationPrinter<DataType> printer;
    printer.print(aggregated_counter_tree.root(), false);
    if (comm.is_root()) {
        std::vector<std::vector<DataType>> const expected_data{{comm.size_signed() * (comm.size_signed() + 1) / 2}};
        std::unordered_map<std::string, AggregatedDataSummary<DataType>> expected_output{
            {"root.measurement:sum",
             AggregatedDataSummary<DataType>{}
                 .set_num_entries(1)
                 .set_num_values_per_entry(1)
                 .set_is_scalar(true)
                 .set_aggregated_data(expected_data)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(CounterTest, gather_aggregation) {
    auto const& comm = comm_world();
    Counter     counter;
    using DataType = decltype(counter)::DataType;
    counter.add("measurement", comm.rank_signed() + 1, {GlobalAggregationMode::gather});
    auto                        aggregated_counter_tree = counter.aggregate();
    ValidationPrinter<DataType> printer;
    printer.print(aggregated_counter_tree.root(), false);
    if (comm.is_root()) {
        std::vector<DataType> values(comm.size());
        std::iota(values.begin(), values.end(), 1);
        std::vector<std::vector<DataType>> const                         expected_data{values};
        std::unordered_map<std::string, AggregatedDataSummary<DataType>> expected_output{
            {"root.measurement:gather",
             AggregatedDataSummary<DataType>{}
                 .set_num_entries(1)
                 .set_num_values_per_entry(comm.size())
                 .set_is_scalar(false)
                 .set_aggregated_data(expected_data)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}

TEST(CounterTest, repeated_add_gather_aggregation) {
    auto const& comm = comm_world();
    Counter     counter;
    using DataType = decltype(counter)::DataType;
    counter.add("measurement", comm.rank_signed() + 1, {GlobalAggregationMode::gather});
    counter.add("measurement", comm.rank_signed() + 1);
    auto                        aggregated_counter_tree = counter.aggregate();
    ValidationPrinter<DataType> printer;
    printer.print(aggregated_counter_tree.root(), false);
    if (comm.is_root()) {
        std::vector<DataType> values;
        for (int i = 1; i <= comm.size_signed(); ++i) {
            values.push_back(2 * i);
        }
        std::vector<std::vector<DataType>> const                         expected_data{values};
        std::unordered_map<std::string, AggregatedDataSummary<DataType>> expected_output{
            {"root.measurement:gather",
             AggregatedDataSummary<DataType>{}
                 .set_num_entries(1)
                 .set_num_values_per_entry(comm.size())
                 .set_is_scalar(false)
                 .set_aggregated_data(expected_data)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}
TEST(CounterTest, repeated_append_gather_aggregation) {
    auto const& comm = comm_world();
    Counter     counter;
    using DataType = decltype(counter)::DataType;
    counter.append("measurement", comm.rank_signed() + 1, {GlobalAggregationMode::gather});
    counter.append("measurement", comm.rank_signed() + 2);
    counter.append("measurement", comm.rank_signed() + 3);
    auto                        aggregated_counter_tree = counter.aggregate();
    ValidationPrinter<DataType> printer;
    printer.print(aggregated_counter_tree.root(), false);
    if (comm.is_root()) {
        std::vector<std::vector<DataType>> expected_data;
        for (int i = 1; i <= 3; ++i) {
            std::vector<DataType> values(comm.size());
            std::iota(values.begin(), values.end(), i);
            expected_data.push_back(values);
        }
        std::unordered_map<std::string, AggregatedDataSummary<DataType>> expected_output{
            {"root.measurement:gather",
             AggregatedDataSummary<DataType>{}
                 .set_num_entries(3)
                 .set_num_values_per_entry(comm.size())
                 .set_is_scalar(false)
                 .set_aggregated_data(expected_data)}};
        EXPECT_EQ(printer.output, expected_output);
    }
}
