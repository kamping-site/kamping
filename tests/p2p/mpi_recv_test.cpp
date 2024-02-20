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

#include "../test_assertions.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/has_member.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/recv.hpp"

using namespace kamping;
using namespace ::testing;

KAMPING_MAKE_HAS_MEMBER(extract_recv_count)
KAMPING_MAKE_HAS_MEMBER(extract_status)
KAMPING_MAKE_HAS_MEMBER(extract_recv_buffer)
KAMPING_MAKE_HAS_MEMBER(extract_recv_type)

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
            auto             result = comm.recv(
                recv_buf<kamping::BufferResizePolicy::resize_to_fit>(message),
                status_out(),
                recv_type_out(),
                recv_count_out()
            );
            EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
            EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
            EXPECT_TRUE(has_member_extract_recv_type_v<decltype(result)>);
            auto status = result.extract_status();
            auto source = status.source();
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(result.extract_recv_type(), MPI_INT);
            EXPECT_EQ(message.size(), source);
            EXPECT_EQ(result.extract_recv_count(), source);
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
            auto             result = comm.recv(
                source(other),
                recv_buf<kamping::BufferResizePolicy::resize_to_fit>(message),
                status_out(),
                recv_count_out()
            );
            EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
            EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
            auto status = result.extract_status();
            auto source = status.source();
            EXPECT_EQ(source, other);
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(message.size(), source);
            EXPECT_EQ(result.extract_recv_count(), source);
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
            auto             result = comm.recv(
                source(other),
                tag(asserting_cast<int>(other)),
                recv_buf<kamping::BufferResizePolicy::resize_to_fit>(message),
                status_out(),
                recv_count_out()
            );
            EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
            EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
            auto status = result.extract_status();
            auto source = status.source();
            EXPECT_EQ(source, other);
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(message.size(), source);
            EXPECT_EQ(result.extract_recv_count(), source);
            EXPECT_EQ(message, std::vector(source, 42));
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_vector_with_explicit_size_resize_to_fit) {
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
        auto result = comm.recv(recv_buf<BufferResizePolicy::resize_to_fit>(message), recv_count(5), status_out());
        EXPECT_FALSE(has_member_extract_recv_count_v<decltype(result)>);
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

TEST_F(RecvTest, recv_vector_with_explicit_size_no_resize_big_enough) {
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
        std::vector<int> message(8, -1);
        EXPECT_EQ(probe_counter, 0);
        auto result = comm.recv(recv_buf<BufferResizePolicy::no_resize>(message), recv_count(5), status_out());
        EXPECT_FALSE(has_member_extract_recv_count_v<decltype(result)>);
        EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
        auto status = result.extract_status();
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 0);
        EXPECT_EQ(status.source(), comm.root());
        EXPECT_EQ(status.tag(), 0);
        EXPECT_EQ(status.count<int>(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5, -1, -1, -1}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
TEST_F(RecvTest, recv_vector_with_explicit_size_no_resize_too_small) {
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
        std::vector<int> message(1);
        EXPECT_EQ(probe_counter, 0);
        EXPECT_KASSERT_FAILS(
            { comm.recv(recv_buf<BufferResizePolicy::no_resize>(message), recv_count(5), status_out()); },
            "Recv buffer is not large enough to hold all received elements."
        );
        // actually receive it to clean up.
        message.resize(5);
        comm.recv(recv_buf<BufferResizePolicy::no_resize>(message), recv_count(5), status_out());
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}
#endif

TEST_F(RecvTest, recv_vector_with_explicit_size_grow_only_big_enough) {
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
        std::vector<int> message(8, -1);
        EXPECT_EQ(probe_counter, 0);
        auto result = comm.recv(recv_buf<BufferResizePolicy::grow_only>(message), recv_count(5), status_out());
        EXPECT_FALSE(has_member_extract_recv_count_v<decltype(result)>);
        EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
        auto status = result.extract_status();
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 0);
        EXPECT_EQ(status.source(), comm.root());
        EXPECT_EQ(status.tag(), 0);
        EXPECT_EQ(status.count<int>(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5, -1, -1, -1}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_vector_with_explicit_size_grow_only_too_small) {
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
        std::vector<int> message(3, -1);
        EXPECT_EQ(probe_counter, 0);
        auto result = comm.recv(recv_buf<BufferResizePolicy::grow_only>(message), recv_count(5), status_out());
        EXPECT_FALSE(has_member_extract_recv_count_v<decltype(result)>);
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
        auto result = comm.recv(
            recv_buf<kamping::BufferResizePolicy::resize_to_fit>(message),
            status_out(recv_status),
            recv_count_out()
        );
        EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_status_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
        EXPECT_EQ(recv_status.source(), comm.root());
        EXPECT_EQ(recv_status.tag(), 0);
        EXPECT_EQ(recv_status.count<int>(), 5);
        EXPECT_EQ(result.extract_recv_count(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_default_custom_container_without_recv_buf) {
    Communicator<::testing::OwnContainer> comm;
    std::vector                           v{1, 2, 3, 4, 5};
    MPI_Request                           req = MPI_REQUEST_NULL;
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
        auto result = comm.recv<int>(recv_count_out());
        EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_status_v<decltype(result)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result)>);
        ::testing::OwnContainer<int> message = result.extract_recv_buffer();
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 1);
        EXPECT_EQ(result.extract_recv_count(), 5);
        EXPECT_EQ(message, ::testing::OwnContainer<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_from_proc_null) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    auto         result = comm.recv(source(rank::null), recv_buf(v), status_out(), recv_count_out());
    EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
    auto   status     = result.extract_status();
    size_t recv_count = static_cast<size_t>(result.extract_recv_count());
    // recv did not touch the buffer
    EXPECT_EQ(v.size(), 5);
    EXPECT_EQ(v, std::vector({1, 2, 3, 4, 5}));
    EXPECT_EQ(status.source_signed(), MPI_PROC_NULL);
    EXPECT_EQ(status.tag(), MPI_ANY_TAG);
    EXPECT_EQ(status.template count<int>(), 0);
    EXPECT_EQ(recv_count, 0);
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST_F(RecvTest, recv_from_invalid_tag) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    EXPECT_KASSERT_FAILS({ comm.recv(recv_buf(v), status_out(), tag(-1)); }, "invalid tag");
}
#endif

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST_F(RecvTest, recv_from_invalid_tag_with_explicit_recv_count) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    EXPECT_KASSERT_FAILS({ comm.recv(recv_buf(v), status_out(), tag(-1), recv_count(1)); }, "invalid tag");
}
#endif

TEST_F(RecvTest, recv_single_int_from_arbitrary_source) {
    Communicator comm;
    int          message = comm.rank_signed();
    MPI_Request  req;
    // Each rank sends a message with its rank as tag to rank 0.
    MPI_Issend(
        &message,                // send_buf
        1,                       // send_count
        MPI_INT,                 // send_type
        0,                       // destination
        comm.rank_signed(),      // tag
        comm.mpi_communicator(), // comm
        &req                     // request
    );
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            Status recv_status;
            int    received_message = comm.recv_single<int>(status_out(recv_status));
            int    source           = recv_status.source_signed();
            EXPECT_EQ(recv_status.tag(), source);
            EXPECT_EQ(recv_status.count<int>(), 1);
            EXPECT_EQ(received_message, source);
        }
    }
    EXPECT_EQ(probe_counter, 0);
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_single_int_from_explicit_source) {
    Communicator comm;
    int          message = comm.rank_signed();
    MPI_Request  req;
    // Each rank sends a message with its rank as tag to rank 0.
    MPI_Issend(
        &message,                // send_buf
        1,                       // send_count
        MPI_INT,                 // send_type
        0,                       // destination
        comm.rank_signed(),      // tag
        comm.mpi_communicator(), // comm
        &req                     // request
    );
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            Status recv_status;
            int    received_message = comm.recv_single<int>(source(other), status_out(recv_status));
            int    source           = recv_status.source_signed();
            EXPECT_EQ(source, other);
            EXPECT_EQ(recv_status.tag(), source);
            EXPECT_EQ(recv_status.count<int>(), 1);
            EXPECT_EQ(received_message, source);
        }
    }
    EXPECT_EQ(probe_counter, 0);
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_single_int_from_explicit_source_and_explicit_tag) {
    Communicator comm;
    int          message = comm.rank_signed();
    MPI_Request  req;
    // Each rank sends a message with its rank as tag to rank 0.
    MPI_Issend(
        &message,                // send_buf
        1,                       // send_count
        MPI_INT,                 // send_type
        0,                       // destination
        comm.rank_signed(),      // tag
        comm.mpi_communicator(), // comm
        &req                     // request
    );
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            Status recv_status;
            int    received_message =
                comm.recv_single<int>(source(other), tag(static_cast<int>(other)), status_out(recv_status));
            int source = recv_status.source_signed();
            EXPECT_EQ(source, other);
            EXPECT_EQ(recv_status.tag(), source);
            EXPECT_EQ(recv_status.count<int>(), 1);
            EXPECT_EQ(received_message, source);
        }
    }
    EXPECT_EQ(probe_counter, 0);
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, recv_single_int_from_explicit_source_and_explicit_ignore_status) {
    Communicator comm;
    int          message = comm.rank_signed();
    MPI_Request  req;
    // Each rank sends a message with its rank as tag to rank 0.
    MPI_Issend(
        &message,                // send_buf
        1,                       // send_count
        MPI_INT,                 // send_type
        0,                       // destination
        comm.rank_signed(),      // tag
        comm.mpi_communicator(), // comm
        &req                     // request
    );
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            int received_message = comm.recv_single<int>(source(other), tag(static_cast<int>(other)));
            EXPECT_EQ(received_message, other);
        }
    }
    EXPECT_EQ(probe_counter, 0);
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST_F(RecvTest, recv_single_int_from_invalid_tag) {
    Communicator comm;
    EXPECT_KASSERT_FAILS({ comm.recv_single<int>(tag(-1)); }, "invalid tag");
}
#endif

TEST_F(RecvTest, recv_type_is_out_param) {
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
        MPI_Datatype recv_type;
        for (size_t other = 0; other < comm.size(); other++) {
            std::vector<int> message;
            auto             result = comm.recv(
                recv_buf<kamping::BufferResizePolicy::resize_to_fit>(message),
                status_out(),
                recv_type_out(recv_type),
                recv_count_out()
            );
            auto status = result.extract_status();
            auto source = status.source();
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(recv_type, MPI_INT);
            EXPECT_EQ(message.size(), source);
            EXPECT_EQ(result.extract_recv_count(), source);
            EXPECT_EQ(message, std::vector(source, 42));
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, non_trivial_recv_type) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;
    comm.barrier();
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
        // ranks are received with padding
        MPI_Datatype int_padding_padding = MPI_INT_padding_padding();
        MPI_Type_commit(&int_padding_padding);
        for (size_t other = 0; other < comm.size(); other++) {
            int const        default_init = -1;
            std::vector<int> message(3 * other, default_init);
            auto             result =
                comm.recv(recv_buf<no_resize>(message), status_out(), source(other), recv_type(int_padding_padding));
            auto status = result.extract_status();
            auto source = status.source();
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(message.size(), 3 * source);
            // EXPECT_EQ(result.extract_recv_count(), source);
            for (size_t i = 0; i < other; ++i) {
                EXPECT_EQ(message[3 * i], 42);
                EXPECT_EQ(message[3 * i + 1], default_init);
                EXPECT_EQ(message[3 * i + 2], default_init);
            }
        }
        MPI_Type_free(&int_padding_padding);
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    comm.barrier();
}

TEST_F(RecvTest, structured_binding_explicit_recv_buf) {
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
            auto [status, recv_count] = comm.recv(
                source(other),
                recv_buf<kamping::BufferResizePolicy::resize_to_fit>(message),
                status_out(),
                recv_count_out()
            );
            auto source = status.source();
            EXPECT_EQ(source, other);
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(message.size(), source);
            EXPECT_EQ(recv_count, source);
            EXPECT_EQ(message, std::vector(source, 42));
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, structured_binding_explicit_owning_recv_buf) {
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
            std::vector<int> tmp;
            auto [status, recv_count, msg] = comm.recv(
                source(other),
                status_out(),
                recv_count_out(),
                recv_buf<kamping::BufferResizePolicy::resize_to_fit>(alloc_container_of<int>)
            );
            auto source = status.source();
            EXPECT_EQ(source, other);
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(msg.size(), source);
            EXPECT_EQ(recv_count, source);
            EXPECT_EQ(msg, std::vector(source, 42));
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(RecvTest, structured_binding_implicit_recv_buf) {
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
            std::vector<int> tmp;
            auto [msg, status, recv_count] = comm.recv<int>(source(other), status_out(), recv_count_out());
            auto source                    = status.source();
            EXPECT_EQ(source, other);
            EXPECT_EQ(status.tag(), source);
            EXPECT_EQ(status.count<int>(), source);
            EXPECT_EQ(msg.size(), source);
            EXPECT_EQ(recv_count, source);
            EXPECT_EQ(msg, std::vector(source, 42));
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}
