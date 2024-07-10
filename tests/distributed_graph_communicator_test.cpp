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

struct DistributedGraphCommunicatorTest : Test {
    void SetUp() override {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int  flag;
        int* value;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &flag);
        EXPECT_TRUE(flag);
        mpi_tag_ub = *value;
    }

    int rank;
    int size;
    int mpi_tag_ub;
};

TEST_F(DistributedGraphCommunicatorTest, empty_communication_graph) {
    Communicator comm;

    DistributedCommunicationGraph<> comm_graph{};
    DistributedGraphCommunicator    graph_comm(comm, comm_graph);

    EXPECT_EQ(graph_comm.compare(kamping::comm_world()), CommunicatorComparisonResult::congruent);
    EXPECT_EQ(graph_comm.rank(), rank);
    EXPECT_EQ(graph_comm.rank_signed(), rank);
    EXPECT_EQ(graph_comm.size_signed(), size);
    EXPECT_EQ(graph_comm.size(), size);
    EXPECT_EQ(graph_comm.root(), 0);
    EXPECT_EQ(graph_comm.root_signed(), 0);
    EXPECT_FALSE(graph_comm.is_weighted());
    EXPECT_EQ(graph_comm.in_degree(), 0);
    EXPECT_EQ(graph_comm.in_degree_signed(), 0);
    EXPECT_EQ(graph_comm.out_degree(), 0);
    EXPECT_EQ(graph_comm.out_degree_signed(), 0);
}

TEST_F(DistributedGraphCommunicatorTest, basics_for_edge_to_predecessor_and_successor_rank) {
    Communicator                    comm;
    std::vector<size_t>             edges{comm.rank_shifted_cyclic(-1), comm.rank_shifted_cyclic(1)};
    DistributedCommunicationGraph<> input_comm_graph(edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    EXPECT_EQ(graph_comm.compare(kamping::comm_world()), CommunicatorComparisonResult::congruent);
    EXPECT_EQ(graph_comm.rank(), rank);
    EXPECT_EQ(graph_comm.rank_signed(), rank);
    EXPECT_EQ(graph_comm.size_signed(), size);
    EXPECT_EQ(graph_comm.size(), size);
    EXPECT_EQ(graph_comm.root(), 0);
    EXPECT_EQ(graph_comm.root_signed(), 0);
    EXPECT_FALSE(graph_comm.is_weighted());
    EXPECT_EQ(graph_comm.in_degree(), 2);
    EXPECT_EQ(graph_comm.in_degree_signed(), 2);
    EXPECT_EQ(graph_comm.out_degree(), 2);
    EXPECT_EQ(graph_comm.out_degree_signed(), 2);
}

TEST_F(DistributedGraphCommunicatorTest, get_communication_graph_for_edge_to_predecessor_and_successor_rank) {
    Communicator                    comm;
    std::vector<size_t>             edges{comm.rank_shifted_cyclic(-1), comm.rank_shifted_cyclic(1)};
    DistributedCommunicationGraph<> input_comm_graph(edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    auto const comm_graph      = graph_comm.get_communication_graph();
    auto const comm_graph_view = comm_graph.get_view();
    EXPECT_TRUE(are_equal(input_comm_graph.get_view(), comm_graph_view));
    EXPECT_FALSE(comm_graph_view.is_weighted());
    EXPECT_EQ(comm_graph_view.in_degree(), 2);
    EXPECT_EQ(comm_graph_view.out_degree(), 2);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(comm.rank_shifted_cyclic(-1), comm.rank_shifted_cyclic(1)));
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(comm.rank_shifted_cyclic(-1), comm.rank_shifted_cyclic(1)));
}

TEST_F(DistributedGraphCommunicatorTest, out_edge_to_successor_rank) {
    Communicator                    comm;
    std::vector<size_t>             out_edges{comm.rank_shifted_cyclic(1)};
    std::vector<size_t>             in_edges{comm.rank_shifted_cyclic(-1)};
    DistributedCommunicationGraph<> input_comm_graph(out_edges, in_edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    EXPECT_EQ(graph_comm.compare(kamping::comm_world()), CommunicatorComparisonResult::congruent);
    EXPECT_EQ(graph_comm.rank(), rank);
    EXPECT_EQ(graph_comm.rank_signed(), rank);
    EXPECT_EQ(graph_comm.size_signed(), size);
    EXPECT_EQ(graph_comm.size(), size);
    EXPECT_EQ(graph_comm.root(), 0);
    EXPECT_EQ(graph_comm.root_signed(), 0);
    EXPECT_FALSE(graph_comm.is_weighted());
    EXPECT_EQ(graph_comm.in_degree(), 1);
    EXPECT_EQ(graph_comm.in_degree_signed(), 1);
    EXPECT_EQ(graph_comm.out_degree(), 1);
    EXPECT_EQ(graph_comm.out_degree_signed(), 1);
}

TEST_F(DistributedGraphCommunicatorTest, get_communication_graph_for_edge_to_successor_rank_and_oneself) {
    Communicator                    comm;
    std::vector<size_t>             in_edges{comm.rank_shifted_cyclic(-1), comm.rank()};
    std::vector<size_t>             out_edges{comm.rank_shifted_cyclic(1), comm.rank()};
    DistributedCommunicationGraph<> input_comm_graph(in_edges, out_edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    auto const comm_graph      = graph_comm.get_communication_graph();
    auto const comm_graph_view = comm_graph.get_view();
    EXPECT_TRUE(are_equal(input_comm_graph.get_view(), comm_graph_view));
    EXPECT_FALSE(comm_graph_view.is_weighted());
    EXPECT_EQ(comm_graph_view.in_degree(), 2);
    EXPECT_EQ(comm_graph_view.out_degree(), 2);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(comm.rank_shifted_cyclic(-1), comm.rank()));
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(comm.rank_shifted_cyclic(1), comm.rank()));
}

TEST_F(DistributedGraphCommunicatorTest, basics_for_edge_to_successor_rank_and_oneself_with_weights) {
    Communicator                        comm;
    std::vector<std::pair<size_t, int>> in_edges{{comm.rank_shifted_cyclic(-1), 42}, {comm.rank(), 0}};
    std::vector<std::pair<size_t, int>> out_edges{{comm.rank_shifted_cyclic(1), 42}, {comm.rank(), 0}};
    DistributedCommunicationGraph<>     input_comm_graph(in_edges, out_edges);
    DistributedGraphCommunicator        graph_comm(comm, input_comm_graph);

    EXPECT_EQ(graph_comm.compare(kamping::comm_world()), CommunicatorComparisonResult::congruent);
    EXPECT_EQ(graph_comm.rank(), rank);
    EXPECT_EQ(graph_comm.rank_signed(), rank);
    EXPECT_EQ(graph_comm.size_signed(), size);
    EXPECT_EQ(graph_comm.size(), size);
    EXPECT_EQ(graph_comm.root(), 0);
    EXPECT_EQ(graph_comm.root_signed(), 0);
    EXPECT_TRUE(graph_comm.is_weighted());
    EXPECT_EQ(graph_comm.in_degree(), 2);
    EXPECT_EQ(graph_comm.in_degree_signed(), 2);
    EXPECT_EQ(graph_comm.out_degree(), 2);
    EXPECT_EQ(graph_comm.out_degree_signed(), 2);
}

TEST_F(DistributedGraphCommunicatorTest, get_communication_graph_for_edge_to_successor_rank_and_oneself_with_weights) {
    Communicator                        comm;
    std::vector<std::pair<size_t, int>> in_edges{{comm.rank_shifted_cyclic(-1), 42}, {comm.rank(), 0}};
    std::vector<std::pair<size_t, int>> out_edges{{comm.rank_shifted_cyclic(1), 42}, {comm.rank(), 0}};

    DistributedCommunicationGraph<> input_comm_graph(in_edges, out_edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    auto const comm_graph      = graph_comm.get_communication_graph();
    auto const comm_graph_view = comm_graph.get_view();
    EXPECT_TRUE(are_equal(input_comm_graph.get_view(), comm_graph_view));
    EXPECT_TRUE(comm_graph_view.is_weighted());
    EXPECT_EQ(comm_graph_view.in_degree(), 2);
    EXPECT_EQ(comm_graph_view.out_degree(), 2);
    EXPECT_THAT(comm_graph_view.in_ranks(), ElementsAre(comm.rank_shifted_cyclic(-1), comm.rank()));
    EXPECT_THAT(comm_graph_view.out_ranks(), ElementsAre(comm.rank_shifted_cyclic(1), comm.rank()));
    EXPECT_THAT(comm_graph_view.in_weights().value(), ElementsAre(42, 0));
    EXPECT_THAT(comm_graph_view.out_weights().value(), ElementsAre(42, 0));
}

TEST_F(DistributedGraphCommunicatorTest, root_to_all_others_from_graph_view) {
    Communicator        comm;
    std::vector<size_t> in_edges{comm.root()};
    std::vector<size_t> out_edges;
    if (comm.is_root()) {
        out_edges.resize(comm.size());
        std::iota(out_edges.begin(), out_edges.end(), 0);
    }
    auto const expected_out_edges = out_edges;

    DistributedCommunicationGraph<> input_comm_graph(in_edges, out_edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    auto const comm_graph      = graph_comm.get_communication_graph();
    auto const comm_graph_view = comm_graph.get_view();
    EXPECT_TRUE(are_equal(input_comm_graph.get_view(), comm_graph_view));
    EXPECT_FALSE(comm_graph_view.is_weighted());
    EXPECT_FALSE(graph_comm.is_weighted());
    EXPECT_EQ(comm_graph_view.in_degree(), 1);
    if (comm.is_root()) {
        EXPECT_EQ(comm_graph_view.out_degree(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(comm_graph_view.out_ranks()[i], i);
        }
    } else {
        EXPECT_EQ(comm_graph_view.out_degree(), comm.root());
    }
}
