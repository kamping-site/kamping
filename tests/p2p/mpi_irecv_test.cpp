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

#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/has_member.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/irecv.hpp"

using namespace kamping;

KAMPING_MAKE_HAS_MEMBER(extract_recv_counts)
KAMPING_MAKE_HAS_MEMBER(extract_status)
KAMPING_MAKE_HAS_MEMBER(extract_recv_buffer)

static size_t call_hierarchy_level = 0;
static size_t probe_counter        = 0;

// call-counting wrapper for probe
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status* status) {
    call_hierarchy_level++;
    auto errcode = PMPI_Probe(source, tag, comm, status);
    // a probe call may call another operation in its implementation
    // this ensures that we only count the top level probe operation
    if (call_hierarchy_level == 1) {
        probe_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

class IrecvTest : public ::testing::Test {
    void SetUp() override {
        call_hierarchy_level = 0;
        probe_counter        = 0;
        // this makes sure that messages don't spill from other tests
        MPI_Barrier(MPI_COMM_WORLD);
    }
    void TearDown() override {
        // this makes sure that messages don't spill to other tests
        MPI_Barrier(MPI_COMM_WORLD);
        call_hierarchy_level = 0;
        probe_counter        = 0;
    }
};

TEST_F(IrecvTest, recv_from_proc_null) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    auto         nonblocking_result = comm.irecv(source(rank::null), recv_buf(v));
    auto         status             = nonblocking_result.test(status_out());
    while (!nonblocking_result.test(status_out())) {
    }
    nonblocking_result.test();
    auto result = result.wait();
    EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result)>);
    auto   status     = result.extract_status();
    size_t recv_count = static_cast<size_t>(result.extract_recv_counts());
    // recv did not touch the buffer
    EXPECT_EQ(v.size(), 5);
    EXPECT_EQ(v, std::vector({1, 2, 3, 4, 5}));
    EXPECT_EQ(status.source_signed(), MPI_PROC_NULL);
    EXPECT_EQ(status.tag(), MPI_ANY_TAG);
    EXPECT_EQ(status.template count<int>(), 0);
    EXPECT_EQ(recv_count, 0);
}
