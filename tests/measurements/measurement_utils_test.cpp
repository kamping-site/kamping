// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#include <algorithm>
#include <optional>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/measurements/aggregated_tree_node.hpp"
#include "kamping/measurements/internal/measurement_utils.hpp"

using namespace kamping::measurements;
using namespace kamping::measurements::internal;

struct DummyNode : public TreeNode<DummyNode> {
    using TreeNode<DummyNode>::TreeNode; // make base class constructors available
};

TEST(TreeNodeTest, node_construction) {
    using namespace kamping::measurements::internal;
    {
        DummyNode root;
        EXPECT_EQ(root.name(), "");
        EXPECT_EQ(root.parent_ptr(), nullptr);
    }
    {
        DummyNode root("root");
        EXPECT_EQ(root.name(), "root");
        EXPECT_EQ(root.parent_ptr(), nullptr);
    }
    {
        DummyNode root;
        DummyNode child("root", &root);

        EXPECT_EQ(child.name(), "root");
        EXPECT_EQ(child.parent_ptr(), &root);
    }
}

TEST(TreeNodeTest, find_or_insert_basic_tree_construction) {
    using namespace kamping::measurements::internal;
    DummyNode root("root");
    auto&     child1  = root.find_or_insert("child1");
    auto&     child2  = root.find_or_insert("child2");
    auto&     child11 = child1.find_or_insert("child11");
    auto&     child12 = child1.find_or_insert("child12");
    // test find part of find_or_insert
    EXPECT_EQ(&root.find_or_insert("child1"), &child1);
    EXPECT_EQ(&root.find_or_insert("child2"), &child2);
    EXPECT_EQ(&child1.find_or_insert("child11"), &child11);
    EXPECT_EQ(&child1.find_or_insert("child12"), &child12);
}

TEST(TreeNodeTest, find_or_insert_basic_navigation_structure) {
    using namespace kamping::measurements::internal;
    DummyNode root("root");
    auto&     child1  = root.find_or_insert("child1");
    auto&     child2  = root.find_or_insert("child2");
    auto&     child11 = child1.find_or_insert("child11");
    auto&     child12 = child1.find_or_insert("child12");

    auto contains_child = [](DummyNode& node, DummyNode& child) {
        auto it = std::find_if(node.children().begin(), node.children().end(), [&](const auto& cur_child) {
            return cur_child.get() == &child;
        });
        return it != node.children().end();
    };
    // check that children and parent ptr are correct
    EXPECT_EQ(root.parent_ptr(), nullptr);
    EXPECT_EQ(root.children().size(), 2u);
    EXPECT_TRUE(contains_child(root, child1));
    EXPECT_TRUE(contains_child(root, child2));

    EXPECT_EQ(child1.parent_ptr(), &root);
    EXPECT_EQ(child1.children().size(), 2u);
    EXPECT_TRUE(contains_child(child1, child11));
    EXPECT_TRUE(contains_child(child1, child12));

    EXPECT_EQ(child2.parent_ptr(), &root);
    EXPECT_EQ(child2.children().size(), 0u);

    EXPECT_EQ(child11.parent_ptr(), &child1);
    EXPECT_EQ(child11.children().size(), 0u);

    EXPECT_EQ(child12.parent_ptr(), &child1);
    EXPECT_EQ(child12.children().size(), 0u);
}

TEST(MeasurementUtilsTest, get_string_for_aggreation_operation_max) {
    EXPECT_EQ(get_string(GlobalAggregationMode::max), "max");
}

TEST(MeasurementUtilsTest, get_string_for_aggreation_operation_min) {
    EXPECT_EQ(get_string(GlobalAggregationMode::min), "min");
}

TEST(MeasurementUtilsTest, get_string_for_aggreation_operation_sum) {
    EXPECT_EQ(get_string(GlobalAggregationMode::sum), "sum");
}

TEST(MeasurementUtilsTest, get_string_for_aggreation_operation_gather) {
    EXPECT_EQ(get_string(GlobalAggregationMode::gather), "gather");
}

TEST(MaxTest, compute_basics) {
    std::vector<int> vec;
    EXPECT_EQ(Max::compute(vec), std::nullopt);
    vec      = {5, 1, 99};
    auto res = Max::compute(vec);
    EXPECT_TRUE(res);
    EXPECT_EQ(res.value(), 99);
}

TEST(MinTest, compute_basics) {
    std::vector<int> vec;
    EXPECT_EQ(Min::compute(vec), std::nullopt);
    vec      = {5, 1, 99};
    auto res = Min::compute(vec);
    EXPECT_TRUE(res);
    EXPECT_EQ(res.value(), 1);
}

TEST(SumTest, compute_basics) {
    std::vector<int> vec;
    EXPECT_EQ(Sum::compute(vec), std::nullopt);
    vec      = {5, 1, 99};
    auto res = Sum::compute(vec);
    EXPECT_TRUE(res);
    EXPECT_EQ(res.value(), 105);
}

TEST(GatherTest, compute_basics) {
    std::vector<int> vec;
    EXPECT_EQ(Gather::compute(vec), vec);
    vec = {5, 1, 99};
    EXPECT_EQ(Gather::compute(vec), vec);
}

TEST(TreeNodeTest, aggregate_measurements_locally_basic_appending) {
    TimerTreeNode<int, int> node;
    int const               duration1 = 2;
    int const               duration2 = 1;
    int const               duration3 = 3;
    EXPECT_EQ(node.measurements().size(), 0u);
    node.aggregate_measurements_locally(duration1, LocalAggregationMode::append);
    EXPECT_EQ(node.measurements(), std::vector<int>{duration1});
    node.aggregate_measurements_locally(duration2, LocalAggregationMode::append);
    node.aggregate_measurements_locally(duration3, LocalAggregationMode::append);
    EXPECT_EQ(node.measurements(), (std::vector<int>{duration1, duration2, duration3}));
}

TEST(TimerTreeNodeTest, aggregate_measurements_locally_basic_accumulate) {
    TimerTreeNode<int, int> node;
    int const               duration1 = 2;
    int const               duration2 = 1;
    int const               duration3 = 3;
    EXPECT_EQ(node.measurements().size(), 0u);
    node.aggregate_measurements_locally(duration1, LocalAggregationMode::accumulate);
    EXPECT_EQ(node.measurements(), std::vector<int>{duration1});
    node.aggregate_measurements_locally(duration2, LocalAggregationMode::accumulate);
    node.aggregate_measurements_locally(duration3, LocalAggregationMode::accumulate);
    EXPECT_EQ(node.measurements(), (std::vector<int>{duration1 + duration2 + duration3}));
}

TEST(TimerTreeNodeTest, aggregate_measurements_locally_basic_interleaved) {
    TimerTreeNode<int, int> node;
    int const               duration1 = 2;
    int const               duration2 = 1;
    int const               duration3 = 3;
    EXPECT_EQ(node.measurements().size(), 0u);
    node.aggregate_measurements_locally(duration1, LocalAggregationMode::accumulate);
    EXPECT_EQ(node.measurements(), std::vector<int>{duration1});
    node.aggregate_measurements_locally(duration2, LocalAggregationMode::append);
    node.aggregate_measurements_locally(duration3, LocalAggregationMode::accumulate);
    EXPECT_EQ(node.measurements(), (std::vector<int>{duration1, duration2 + duration3}));
}

TEST(TreeTest, constructor) {
    Tree<TimerTreeNode<int, std::size_t>> timer_tree;
    EXPECT_EQ(timer_tree.current_node, &timer_tree.root);
    EXPECT_EQ(timer_tree.root.name(), "root");
    EXPECT_EQ(timer_tree.root.children().size(), 0u);
    EXPECT_EQ(timer_tree.root.parent_ptr(), &timer_tree.root);
}

TEST(AggregatedTreeNodeTest, add_one_aggregation_operation) {
    AggregatedTreeNode<double> node;
    double const               value1 = 5.0;
    std::vector<double> const  value2{6.0, 6.0};
    auto                       operation = GlobalAggregationMode::max;
    // add first result of aggregation op operation
    node.add(operation, std::optional<double>{value1});
    EXPECT_EQ(node.aggregated_data().size(), 1u);
    // add second result of aggregation op operation (which is empty and should result in a noop)
    node.add(operation, std::optional<double>{});
    EXPECT_EQ(node.aggregated_data().size(), 1u);
    // add third result of aggregation op operation which is a list
    node.add(operation, value2);
    EXPECT_EQ(node.aggregated_data().size(), 1u);

    auto it = node.aggregated_data().find(operation);
    ASSERT_NE(it, node.aggregated_data().end());
    auto&                                  contained_values = it->second;
    std::vector<ScalarOrContainer<double>> expected_values{value1, value2};
    EXPECT_EQ(contained_values, expected_values);
}

TEST(AggregatedTreeNodeTest, add_multiple_aggregation_operation) {
    AggregatedTreeNode<double> node;
    double const               value1 = 5.0;
    std::vector<double> const  value2{6.0, 6.0};
    auto                       operation1 = GlobalAggregationMode::max;
    auto                       operation2 = GlobalAggregationMode::min;
    auto                       operation3 = GlobalAggregationMode::gather;
    // add result of aggregation op operation1
    node.add(operation1, std::optional<double>{value1});
    EXPECT_EQ(node.aggregated_data().size(), 1u);
    // add second result of aggregation op operation2 (which is empty and should result in a noop)
    node.add(operation2, std::optional<double>{});
    EXPECT_EQ(node.aggregated_data().size(), 1u);
    // add result of aggregation op operation3 which is a list
    node.add(operation3, value2);
    EXPECT_EQ(node.aggregated_data().size(), 2u);
    {
        auto it = node.aggregated_data().find(operation1);
        ASSERT_NE(it, node.aggregated_data().end());
        auto&                                  contained_values = it->second;
        std::vector<ScalarOrContainer<double>> expected_values{value1};
        EXPECT_EQ(contained_values, expected_values);
    }
    {
        auto it = node.aggregated_data().find(operation3);
        ASSERT_NE(it, node.aggregated_data().end());
        auto&                                  contained_values = it->second;
        std::vector<ScalarOrContainer<double>> expected_values{value2};
        EXPECT_EQ(contained_values, expected_values);
    }
}
