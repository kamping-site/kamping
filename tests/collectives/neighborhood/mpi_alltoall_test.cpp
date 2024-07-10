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

#include "../../test_assertions.hpp"

#include <algorithm>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kamping/collectives/neighborhood/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/distributed_graph_communicator.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(NeighborhoodAlltoallTest, single_element_no_receive_buffer_for_edges_to_predecessor_successor) {
    Communicator                    comm;
    std::vector<size_t>             edges{comm.rank_shifted_cyclic(-1), comm.rank_shifted_cyclic(1)};
    DistributedCommunicationGraph<> input_comm_graph(edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    std::vector<size_t> const input = edges;

    auto mpi_result = graph_comm.neighbor_alltoall(send_buf(input), send_count_out(), recv_count_out());

    auto recv_buffer = mpi_result.extract_recv_buffer();
    auto send_count  = mpi_result.extract_send_count();
    auto recv_count  = mpi_result.extract_recv_count();

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_buffer.size(), graph_comm.in_degree());

    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank(), comm.rank()));
}

TEST(NeighborhoodAlltoallTest, single_element_no_receive_buffer_for_edges_to_successor) {
    Communicator                    comm;
    std::vector<size_t>             in_edges{comm.rank_shifted_cyclic(-1)};
    std::vector<size_t>             out_edges{comm.rank_shifted_cyclic(1)};
    DistributedCommunicationGraph<> input_comm_graph(in_edges, out_edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    std::vector<size_t> const input{comm.rank()};

    auto mpi_result = graph_comm.neighbor_alltoall(send_buf(input), send_count_out(), recv_count_out());

    auto recv_buffer = mpi_result.extract_recv_buffer();
    auto send_count  = mpi_result.extract_send_count();
    auto recv_count  = mpi_result.extract_recv_count();

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_buffer.size(), graph_comm.in_degree());

    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank_shifted_cyclic(-1)));
}

TEST(NeighborhoodAlltoallTest, single_element_no_receive_buffer_for_multi_edges_to_successor) {
    Communicator                    comm;
    size_t const                    edge_multiplicity = 3;
    std::vector<size_t> const       in_edges(edge_multiplicity, comm.rank_shifted_cyclic(-1));
    std::vector<size_t> const       out_edges(edge_multiplicity, comm.rank_shifted_cyclic(1));
    DistributedCommunicationGraph<> input_comm_graph(in_edges, out_edges);
    DistributedGraphCommunicator    graph_comm(comm, input_comm_graph);

    std::vector<size_t> const input(edge_multiplicity, comm.rank());

    auto mpi_result = graph_comm.neighbor_alltoall(send_buf(input), send_count_out(), recv_count_out());

    auto recv_buffer = mpi_result.extract_recv_buffer();
    auto send_count  = mpi_result.extract_send_count();
    auto recv_count  = mpi_result.extract_recv_count();

    EXPECT_EQ(send_count, 1);
    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(recv_buffer.size(), graph_comm.in_degree());

    EXPECT_EQ(recv_buffer, in_edges);
}

/// @todo reactivate once the correct handling of send/recv count has been clarified
// TEST(NeighborhoodAlltoallTest, single_element_with_isolated_root) {
//     Communicator comm;
//     // root rank is isolated but all others have edge to successor
//     size_t              predecessor = comm.rank_shifted_cyclic(-1);
//     size_t              successor   = comm.rank_shifted_cyclic(1);
//     std::vector<size_t> in_edges{predecessor};
//     std::vector<size_t> out_edges{successor};
//     if (comm.is_root()) {
//         in_edges.clear();
//         out_edges.clear();
//     }
//     bool is_pred_of_root = comm.root() == successor;
//     bool is_succ_of_root = comm.root() == predecessor;
//     if (is_pred_of_root) {
//         out_edges.clear();
//     }
//     if (is_succ_of_root) {
//         in_edges.clear();
//     }
//     DistributedCommunicationGraph<>         input_comm_graph(in_edges, out_edges);
//     DistributedGraphCommunicator graph_comm(comm, input_comm_graph);
//
//     std::vector<size_t> input{comm.rank()};
//     if (comm.is_root() || is_pred_of_root) {
//         input.clear();
//     }
//
//     if (comm.is_root()) {
//         auto recv_buf = graph_comm.neighbor_alltoall(send_buf(input), send_count(0), recv_count(0));
//         EXPECT_EQ(recv_buf.size(), 0);
//     } else if (is_succ_of_root) {
//         auto [recv_buf, recv_count] = graph_comm.neighbor_alltoall(send_buf(input), send_count(1), recv_count_out());
//         EXPECT_EQ(recv_count, 1);
//         EXPECT_EQ(recv_buf.size(), 0);
//     } else if (is_pred_of_root) {
//         auto mpi_result = graph_comm.neighbor_alltoall(send_buf(input), send_count(1), recv_count(1));
//         EXPECT_THAT(mpi_result, ElementsAre(predecessor));
//     } else {
//         auto [recv_buf, send_count, recv_count] =
//             graph_comm.neighbor_alltoall(send_buf(input), send_count_out(), recv_count_out());
//         EXPECT_EQ(send_count, 1);
//         EXPECT_EQ(recv_count, 1);
//         EXPECT_THAT(recv_buf, ElementsAre(predecessor));
//     }
// }
