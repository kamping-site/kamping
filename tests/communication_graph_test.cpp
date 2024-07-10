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

#include "test_assertions.hpp"

#include "gmock/gmock.h"
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "helpers_for_testing.hpp"
#include "kamping/comm_helper/num_numa_nodes.hpp"
#include "kamping/communicator.hpp"
#include "kamping/distributed_graph_communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(CommunicationGraphTest, empty) {
    DistributedCommunicationGraph<> comm_graph{};
    auto                            comm_graph_view = comm_graph.get_view();
    EXPECT_EQ(comm_graph_view.in_degree(), 0);
    EXPECT_EQ(comm_graph_view.in_ranks().size(), 0);
    EXPECT_EQ(comm_graph_view.in_weights(), std::nullopt);
    EXPECT_EQ(comm_graph_view.out_degree(), 0);
    EXPECT_EQ(comm_graph_view.out_ranks().size(), 0);
    EXPECT_EQ(comm_graph_view.out_weights(), std::nullopt);
    EXPECT_FALSE(comm_graph_view.is_weighted());
}

TEST(CommunicationGraphTest, unweighted_symmetric_edges) {
    std::vector<size_t>             edges{1, 2, 3};
    DistributedCommunicationGraph<> comm_graph{edges};
    auto                            comm_graph_view = comm_graph.get_view();
    EXPECT_EQ(comm_graph_view.in_degree(), 3);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(1, 2, 3));
    EXPECT_EQ(comm_graph_view.in_weights(), std::nullopt);
    EXPECT_EQ(comm_graph_view.out_degree(), 3);
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(1, 2, 3));
    EXPECT_EQ(comm_graph_view.out_weights(), std::nullopt);
    EXPECT_FALSE(comm_graph_view.is_weighted());
}

TEST(CommunicationGraphTest, unweighted_asymmetric_edges) {
    std::vector<size_t>             in_edges{1, 2, 3, 4};
    std::vector<size_t>             out_edges{5, 6, 7};
    DistributedCommunicationGraph<> comm_graph{in_edges, out_edges};
    auto                            comm_graph_view = comm_graph.get_view();
    EXPECT_EQ(comm_graph_view.in_degree(), 4);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(1, 2, 3, 4));
    EXPECT_EQ(comm_graph_view.in_weights(), std::nullopt);
    EXPECT_EQ(comm_graph_view.out_degree(), 3);
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(5, 6, 7));
    EXPECT_EQ(comm_graph_view.out_weights(), std::nullopt);
    EXPECT_FALSE(comm_graph_view.is_weighted());
}

TEST(CommunicationGraphTest, weighted_asymmetric_edges) {
    std::vector<std::pair<size_t, int>> in_edges{{1, 4}, {2, 3}, {3, 2}, {4, 1}};
    std::vector<std::pair<size_t, int>> out_edges{{5, 7}, {6, 6}, {7, 5}};
    DistributedCommunicationGraph<>     comm_graph{in_edges, out_edges};
    auto                                comm_graph_view = comm_graph.get_view();
    EXPECT_EQ(comm_graph_view.in_degree(), 4);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(1, 2, 3, 4));
    EXPECT_THAT(comm_graph_view.in_weights().value(), ElementsAre(4, 3, 2, 1));
    EXPECT_EQ(comm_graph_view.out_degree(), 3);
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(5, 6, 7));
    EXPECT_THAT(comm_graph_view.out_weights().value(), ElementsAre(7, 6, 5));
    EXPECT_TRUE(comm_graph_view.is_weighted());
}

struct OwnEdge {
    int rank;
    int weight;
};

TEST(CommunicationGraphTest, weighted_asymmetric_edges_with_custom_edge_type) {
    std::vector<OwnEdge>            in_edges{{1, 4}, {2, 3}, {3, 2}, {4, 1}};
    std::vector<OwnEdge>            out_edges{{5, 7}, {6, 6}, {7, 5}};
    DistributedCommunicationGraph<> comm_graph{in_edges, out_edges};
    auto                            comm_graph_view = comm_graph.get_view();
    EXPECT_EQ(comm_graph_view.in_degree(), 4);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(1, 2, 3, 4));
    EXPECT_THAT(comm_graph_view.in_weights().value(), ElementsAre(4, 3, 2, 1));
    EXPECT_EQ(comm_graph_view.out_degree(), 3);
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(5, 6, 7));
    EXPECT_THAT(comm_graph_view.out_weights().value(), ElementsAre(7, 6, 5));
    EXPECT_TRUE(comm_graph_view.is_weighted());
}

TEST(CommunicationGraphTest, unweighted_asymmetric_edges_with_move_construction) {
    std::vector<int>                in_edges{1, 2, 3, 4};
    std::vector<int>                out_edges{5, 6, 7};
    DistributedCommunicationGraph<> comm_graph{std::move(in_edges), std::move(out_edges)};
    auto                            comm_graph_view = comm_graph.get_view();
    EXPECT_EQ(comm_graph_view.in_degree(), 4);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(1, 2, 3, 4));
    EXPECT_EQ(comm_graph_view.in_weights(), std::nullopt);
    EXPECT_EQ(comm_graph_view.out_degree(), 3);
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(5, 6, 7));
    EXPECT_EQ(comm_graph_view.out_weights(), std::nullopt);
    EXPECT_FALSE(comm_graph_view.is_weighted());
}

TEST(CommunicationGraphTest, weighted_asymmetric_edges_with_move_construction) {
    std::vector<int>                in_edges{1, 2, 3, 4};
    std::vector<int>                in_weights{4, 3, 2, 1};
    std::vector<int>                out_edges{5, 6, 7};
    std::vector<int>                out_weights{7, 6, 5};
    DistributedCommunicationGraph<> comm_graph{
        std::move(in_edges),
        std::move(out_edges),
        std::move(in_weights),
        std::move(out_weights)};
    auto comm_graph_view = comm_graph.get_view();
    EXPECT_EQ(comm_graph_view.in_degree(), 4);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(1, 2, 3, 4));
    EXPECT_THAT(comm_graph_view.in_weights().value(), ElementsAre(4, 3, 2, 1));
    EXPECT_EQ(comm_graph_view.out_degree(), 3);
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(5, 6, 7));
    EXPECT_THAT(comm_graph_view.out_weights().value(), ElementsAre(7, 6, 5));
    EXPECT_TRUE(comm_graph_view.is_weighted());
}

TEST(CommunicationGraphTest, rank_to_out_edge_mapping_for_unweighted_asymmetric_edges) {
    std::vector<int>                in_edges{1, 2, 3, 4};
    std::vector<int>                out_edges{5, 6, 7};
    DistributedCommunicationGraph<> comm_graph{std::move(in_edges), std::move(out_edges)};
    auto                            mapping = comm_graph.get_rank_to_out_neighbor_idx_mapping();
    EXPECT_EQ(mapping.size(), 3);
    auto it1 = mapping.find(5);
    EXPECT_NE(it1, mapping.end());
    EXPECT_EQ(it1->second, 0);
    auto it2 = mapping.find(6);
    EXPECT_NE(it2, mapping.end());
    EXPECT_EQ(it2->second, 1);
    auto it3 = mapping.find(7);
    EXPECT_NE(it3, mapping.end());
    EXPECT_EQ(it3->second, 2);
}
