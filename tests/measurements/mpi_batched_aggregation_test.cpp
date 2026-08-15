// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/measurements/aggregated_tree_node.hpp"
#include "kamping/measurements/internal/measurement_utils.hpp"
#include "kamping/measurements/measurement_aggregation_definitions.hpp"
#include "measurement_test_helpers.hpp"

using namespace ::kamping;
using namespace ::kamping::measurements;
using namespace ::testing;

namespace {

using DataType        = std::int64_t;
using MeasurementNode = kamping::measurements::internal::CounterTreeNode<DataType>;
using MeasurementTree = kamping::measurements::internal::Tree<MeasurementNode>;
using AggregatedNode  = AggregatedTreeNode<DataType>;

/// @brief Adds a child with the given datapoints and global aggregation operations to the given node.
MeasurementNode& add_node(
    MeasurementNode&                          parent,
    std::string const&                        name,
    std::vector<DataType> const&              datapoints,
    std::vector<GlobalAggregationMode> const& aggregation_modes
) {
    auto& child = parent.find_or_insert(name);
    for (auto const& datapoint: datapoints) {
        child.aggregate_measurements_locally(datapoint, LocalAggregationMode::append);
    }
    child.measurements_aggregation_operations() = aggregation_modes;
    return child;
}

/// @brief Creates a measurement tree containing nodes with multiple datapoints, mixed aggregation operations and
/// nodes without any datapoints. The stored datapoints depend on the given rank.
MeasurementTree create_measurement_tree(std::size_t rank) {
    auto const offset = static_cast<DataType>(rank) * 100;

    MeasurementTree tree;
    auto&           algorithm = add_node(tree.root, "algorithm", {offset + 1}, {GlobalAggregationMode::max});
    add_node(algorithm, "preprocessing", {offset + 2, offset + 3}, {GlobalAggregationMode::min});
    auto& core = add_node(
        algorithm,
        "core",
        {offset + 4, offset + 5, offset + 6},
        {GlobalAggregationMode::max, GlobalAggregationMode::min, GlobalAggregationMode::sum}
    );
    add_node(core, "without_datapoints", {}, {GlobalAggregationMode::sum});
    add_node(core, "subroutine", {offset + 7}, {GlobalAggregationMode::gather});
    add_node(algorithm, "postprocessing", {offset + 8}, {GlobalAggregationMode::sum});
    return tree;
}

/// @brief Collects the names of all nodes of the given aggregated tree in depth-first order.
void collect_node_names(AggregatedNode const& node, std::string const& prefix, std::vector<std::string>& names) {
    std::string const path = prefix.empty() ? node.name() : prefix + "." + node.name();
    names.push_back(path);
    for (auto const& child: node.children()) {
        collect_node_names(*child, path, names);
    }
}

std::vector<std::string> node_names(AggregatedNode const& root) {
    std::vector<std::string> names;
    collect_node_names(root, "", names);
    return names;
}

std::unordered_map<std::string, AggregatedDataSummary<DataType>> summarize(AggregatedNode const& root) {
    ValidationPrinter<DataType> printer;
    printer.print(root, false);
    return printer.output;
}

} // namespace

TEST(BatchedAggregationTest, aggregate_matches_unbatched_aggregation) {
    auto const& comm = comm_world();
    auto const  tree = create_measurement_tree(comm.rank());

    AggregatedTree<DataType> batched(tree.root, comm);
    AggregatedTree<DataType> unbatched(AggregatedTree<DataType>::UnbatchedTag{}, tree.root, comm);

    EXPECT_EQ(node_names(batched.root()), node_names(unbatched.root()));
    if (comm.is_root()) {
        EXPECT_EQ(summarize(batched.root()), summarize(unbatched.root()));
    }
}

TEST(BatchedAggregationTest, aggregate_matches_unbatched_aggregation_with_split_batches) {
    auto const& comm = comm_world();
    auto const  tree = create_measurement_tree(comm.rank());

    AggregatedTree<DataType> unbatched(AggregatedTree<DataType>::UnbatchedTag{}, tree.root, comm);

    // a budget of a single datapoint per rank forces one collective per datapoint, splitting the nodes which store
    // multiple datapoints across batches
    for (std::size_t max_bytes_at_root:
         {std::size_t{1}, sizeof(DataType) * comm.size(), sizeof(DataType) * comm.size() * 2}) {
        AggregatedTree<DataType> batched(tree.root, comm, max_bytes_at_root);
        EXPECT_EQ(node_names(batched.root()), node_names(unbatched.root()));
        if (comm.is_root()) {
            EXPECT_EQ(summarize(batched.root()), summarize(unbatched.root()));
        }
    }
}

TEST(BatchedAggregationTest, aggregate_computes_expected_values) {
    auto const& comm = comm_world();
    auto const  tree = create_measurement_tree(comm.rank());

    AggregatedTree<DataType> batched(tree.root, comm);

    if (!comm.is_root()) {
        return;
    }
    auto const     output         = summarize(batched.root());
    auto const     num_ranks      = static_cast<DataType>(comm.size());
    DataType const largest_offset = 100 * (num_ranks - 1);
    DataType const sum_of_offsets = 100 * num_ranks * (num_ranks - 1) / 2;

    ASSERT_EQ(output.count("root.algorithm.core:max"), 1u);
    EXPECT_EQ(
        output.at("root.algorithm.core:max").aggregated_data,
        (std::vector<std::vector<DataType>>{{largest_offset + 4}, {largest_offset + 5}, {largest_offset + 6}})
    );
    ASSERT_EQ(output.count("root.algorithm.core:min"), 1u);
    EXPECT_EQ(
        output.at("root.algorithm.core:min").aggregated_data,
        (std::vector<std::vector<DataType>>{{4}, {5}, {6}})
    );
    ASSERT_EQ(output.count("root.algorithm.core:sum"), 1u);
    EXPECT_EQ(
        output.at("root.algorithm.core:sum").aggregated_data,
        (std::vector<std::vector<DataType>>{
            {sum_of_offsets + 4 * num_ranks},
            {sum_of_offsets + 5 * num_ranks},
            {sum_of_offsets + 6 * num_ranks}})
    );

    std::vector<DataType> expected_gathered;
    for (DataType rank = 0; rank < num_ranks; ++rank) {
        expected_gathered.push_back(100 * rank + 7);
    }
    ASSERT_EQ(output.count("root.algorithm.core.subroutine:gather"), 1u);
    EXPECT_EQ(
        output.at("root.algorithm.core.subroutine:gather").aggregated_data,
        (std::vector<std::vector<DataType>>{expected_gathered})
    );
}

TEST(BatchedAggregationTest, aggregate_keeps_nodes_without_datapoints) {
    auto const& comm = comm_world();
    auto const  tree = create_measurement_tree(comm.rank());

    AggregatedTree<DataType> batched(tree.root, comm);

    std::vector<std::string> const expected_names{
        "root",
        "root.algorithm",
        "root.algorithm.preprocessing",
        "root.algorithm.core",
        "root.algorithm.core.without_datapoints",
        "root.algorithm.core.subroutine",
        "root.algorithm.postprocessing"};
    EXPECT_EQ(node_names(batched.root()), expected_names);
}

TEST(BatchedAggregationTest, aggregate_empty_tree) {
    auto const&     comm = comm_world();
    MeasurementTree tree;

    AggregatedTree<DataType> batched(tree.root, comm);

    EXPECT_EQ(node_names(batched.root()), std::vector<std::string>{"root"});
    if (comm.is_root()) {
        EXPECT_TRUE(summarize(batched.root()).empty());
    }
}

TEST(BatchedAggregationTest, aggregate_diverged_node_names_throws) {
    auto const& comm = comm_world();
    if (comm.size() < 2) {
        GTEST_SKIP() << "Diverging measurement trees require at least two ranks.";
    }
    MeasurementTree tree;
    add_node(tree.root, comm.is_root() ? "on_root" : "on_other_ranks", {1}, {GlobalAggregationMode::max});

    EXPECT_THROW(AggregatedTree<DataType>(tree.root, comm), kassert::KassertException);
}

TEST(BatchedAggregationTest, aggregate_diverged_dimensions_throws) {
    auto const& comm = comm_world();
    if (comm.size() < 2) {
        GTEST_SKIP() << "Diverging measurement trees require at least two ranks.";
    }
    MeasurementTree tree;
    add_node(
        tree.root,
        "measurement",
        comm.is_root() ? std::vector<DataType>{1} : std::vector<DataType>{1, 2},
        {GlobalAggregationMode::max}
    );

    EXPECT_THROW(AggregatedTree<DataType>(tree.root, comm), kassert::KassertException);
}

TEST(BatchedAggregationTest, aggregate_diverged_node_count_throws) {
    auto const& comm = comm_world();
    if (comm.size() < 2) {
        GTEST_SKIP() << "Diverging measurement trees require at least two ranks.";
    }
    MeasurementTree tree;
    add_node(tree.root, "measurement", {1}, {GlobalAggregationMode::max});
    if (comm.is_root()) {
        add_node(tree.root, "only_on_root", {1}, {GlobalAggregationMode::max});
    }

    EXPECT_THROW(AggregatedTree<DataType>(tree.root, comm), kassert::KassertException);
}
