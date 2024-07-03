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

#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <kassert/kassert.hpp>
#include <mpi.h>

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

TEST_F(DistributedGraphCommunicatorTest, empty_constructor) {
    Communicator comm;
    const size_t prev_rank = (comm.rank() + comm.size() - 1) % comm.size();
    const size_t succ_rank = (comm.rank() + 1) % comm.size();
    std::vector<size_t> edges{prev_rank, succ_rank};
    CommunicationGraph comm_graph(edges);
    DistributedGraphCommunicator graph_comm(comm, comm_graph);
    //Communicator<>* comm_ptr = &graph_comm;
    std::stringstream ss;
    ss << "rank: " << comm.rank() << ": " << graph_comm.in_degree() << "\n";

    auto comm_graph2 = graph_comm.get_communication_graph();
    auto mapping = comm_graph2.get_rank_to_out_edge_idx_mapping();
    if (graph_comm.rank() == 0) {
            ss << "mapping: \n";
        for(const auto& [rank_, idx] : mapping) {
            ss << "rank: " << rank_ << " idx: " << idx << "\n";
        }
    }
    std::cout << ss.str() << std::endl;

    EXPECT_FALSE(true);
    //EXPECT_EQ(comm.mpi_communicator(), MPI_COMM_WORLD);
    //EXPECT_EQ(comm.rank(), rank);
    //EXPECT_EQ(comm.rank_signed(), rank);
    //EXPECT_EQ(comm.size_signed(), size);
    //EXPECT_EQ(comm.size(), size);
    //EXPECT_EQ(comm.root(), 0);
    //EXPECT_EQ(comm.root_signed(), 0);
}
