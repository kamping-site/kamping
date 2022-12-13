// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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
#include "kamping/p2p/recv.hpp"

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

class RecvTest : public ::testing::Test {
    void SetUp() override {
        call_hierarchy_level = 0;
        probe_counter        = 0;
    }
    void TearDown() override {
        call_hierarchy_level = 0;
        probe_counter        = 0;
    }
};

TEST_F(RecvTest, recv_vector_from_arbitrary_source) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;
    // Each rank sends a message with its rank as tag to rank 0.
    // The message has comm.rank() elements.
    MPI_Issend(
        v.data(),                      // send_buf
        asserting_cast<int>(v.size()), // send_count
        MPI_INT,                       // send_type
        0,                             // destination
        comm.rank_signed(),            // tag
        comm.mpi_communicator(),       // comm
        &req                           // request
    );
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            std::vector<int> message;
            auto             result = comm.recv(recv_buf(message), status_out());
            EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result)>);
            EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
            auto status = result.extract_status();
            auto source = status.source();
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(message.size(), source);
            EXPECT_EQ(result.extract_recv_counts(), source);
            EXPECT_EQ(message, std::vector(source, 42));
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_vector_from_explicit_source) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;
    // Each rank sends a message with its rank as tag to rank 0.
    // The message has comm.rank() elements.
    MPI_Issend(
        v.data(),                      // send_buf
        asserting_cast<int>(v.size()), // send_count
        MPI_INT,                       // send_type
        0,                             // destination
        comm.rank_signed(),            // tag
        comm.mpi_communicator(),       // comm
        &req                           // request
    );
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            std::vector<int> message;
            auto             result = comm.recv(source(other), recv_buf(message), status_out());
            EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result)>);
            EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
            auto status = result.extract_status();
            auto source = status.source();
            EXPECT_EQ(source, other);
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(message.size(), source);
            EXPECT_EQ(result.extract_recv_counts(), source);
            EXPECT_EQ(message, std::vector(source, 42));
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_vector_from_explicit_source_and_explicit_tag) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;
    // Each rank sends a message with its rank as tag to rank 0.
    // The message has comm.rank() elements.
    MPI_Issend(
        v.data(),                      // send_buf
        asserting_cast<int>(v.size()), // send_count
        MPI_INT,                       // send_type
        0,                             // destination
        comm.rank_signed(),            // tag
        comm.mpi_communicator(),       // comm
        &req                           // request
    );
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            std::vector<int> message;
            auto result = comm.recv(source(other), tag(asserting_cast<int>(other)), recv_buf(message), status_out());
            EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result)>);
            EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
            auto status = result.extract_status();
            auto source = status.source();
            EXPECT_EQ(source, other);
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(message.size(), source);
            EXPECT_EQ(result.extract_recv_counts(), source);
            EXPECT_EQ(message, std::vector(source, 42));
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_vector_with_explicit_size) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    MPI_Request  req = MPI_REQUEST_NULL;
    if (comm.is_root()) {
        auto other_rank = comm.rank_shifted_cyclic(1);
        MPI_Isend(
            v.data(),                        // send_buf
            asserting_cast<int>(v.size()),   // send_count
            MPI_INT,                         // datatype
            asserting_cast<int>(other_rank), // destination
            0,                               // tag
            comm.mpi_communicator(),         // comm
            &req
        );
    }
    if (comm.rank_shifted_cyclic(-1) == comm.root()) {
        std::vector<int> message;
        EXPECT_EQ(probe_counter, 0);
        auto result = comm.recv(recv_buf(message), recv_counts(5), status_out());
        EXPECT_FALSE(has_member_extract_recv_counts_v<decltype(result)>);
        EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
        auto status = result.extract_status();
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 0);
        EXPECT_EQ(status.source(), comm.root());
        EXPECT_EQ(status.tag(), 0);
        EXPECT_EQ(status.count<int>(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_vector_with_input_status) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    MPI_Request  req = MPI_REQUEST_NULL;
    if (comm.is_root()) {
        auto other_rank = comm.rank_shifted_cyclic(1);
        MPI_Isend(
            v.data(),                        // send_buf
            asserting_cast<int>(v.size()),   // send_count
            MPI_INT,                         // datatype
            asserting_cast<int>(other_rank), // destination
            0,                               // tag
            comm.mpi_communicator(),         // comm
            &req
        );
    }
    if (comm.rank_shifted_cyclic(-1) == comm.root()) {
        std::vector<int> message;
        Status           recv_status;
        // pass status as input parameter
        auto result = comm.recv(recv_buf(message), status(recv_status));
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_status_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
        EXPECT_EQ(recv_status.source(), comm.root());
        EXPECT_EQ(recv_status.tag(), 0);
        EXPECT_EQ(recv_status.count<int>(), 5);
        EXPECT_EQ(result.extract_recv_counts(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_default_custom_container_without_recv_buf) {
    Communicator<testing::OwnContainer> comm;
    std::vector                         v{1, 2, 3, 4, 5};
    MPI_Request                         req = MPI_REQUEST_NULL;
    if (comm.is_root()) {
        auto other_rank = comm.rank_shifted_cyclic(1);
        MPI_Isend(
            v.data(),                        // send_buf
            asserting_cast<int>(v.size()),   // send_count
            MPI_INT,                         // datatype
            asserting_cast<int>(other_rank), // destination
            0,                               // tag
            comm.mpi_communicator(),         // comm
            &req
        );
    }
    if (comm.rank_shifted_cyclic(-1) == comm.root()) {
        EXPECT_EQ(probe_counter, 0);
        auto result = comm.recv<int>();
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_status_v<decltype(result)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result)>);
        testing::OwnContainer<int> message = result.extract_recv_buffer();
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 1);
        EXPECT_EQ(result.extract_recv_counts(), 5);
        EXPECT_EQ(message, testing::OwnContainer<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_from_proc_null) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    auto         result = comm.recv(source(rank::null), recv_buf(v), status_out());
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
