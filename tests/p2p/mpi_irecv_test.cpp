// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/has_member.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/irecv.hpp"

using namespace kamping;

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

TEST_F(IrecvTest, recv_vector_from_arbitrary_source) {
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
            auto [handle, result] =
                comm.irecv(recv_buf<resize_to_fit>(message), recv_type_out(), recv_count_out()).extract();
            auto status = handle.wait(status_out());
            EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
            EXPECT_TRUE(has_member_extract_recv_type_v<decltype(result)>);
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

TEST_F(IrecvTest, recv_vector_from_explicit_source) {
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
            // TODO this should not have to be extracted
            auto [handle, result] =
                comm.irecv(source(other), recv_buf<resize_to_fit>(message), recv_count_out()).extract();
            auto status = handle.wait(status_out());
            EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
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

TEST_F(IrecvTest, recv_vector_from_explicit_source_and_explicit_tag) {
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
            auto [handle, result] = comm.irecv(
                                            source(other),
                                            tag(asserting_cast<int>(other)),
                                            recv_buf<resize_to_fit>(message),
                                            recv_count_out()
            )
                                        .extract();
            auto status = handle.wait(status_out());
            EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
            EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
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

TEST_F(IrecvTest, recv_vector_with_explicit_size_resize_to_fit) {
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
        auto handle = comm.irecv(recv_buf<resize_to_fit>(message), recv_count(5)).extract();
        auto status = handle.wait(status_out());
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 0);
        EXPECT_EQ(status.source(), comm.root());
        EXPECT_EQ(status.tag(), 0);
        EXPECT_EQ(status.count<int>(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IrecvTest, recv_vector_with_explicit_size_no_resize_big_enough) {
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
        auto handle = comm.irecv(recv_buf<no_resize>(message), recv_count(5)).extract();
        auto status = handle.wait(status_out());
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
TEST_F(IrecvTest, recv_vector_with_explicit_size_no_resize_too_small) {
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
            { comm.irecv(recv_buf<no_resize>(message), recv_count(5)); },
            "Recv buffer is not large enough to hold all received elements."
        );
        // actually receive it to clean up.
        message.resize(5);
        comm.irecv(recv_buf<no_resize>(message), recv_count(5)).wait();
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}
#endif

TEST_F(IrecvTest, recv_vector_with_explicit_size_grow_only_big_enough) {
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
        auto handle = comm.irecv(recv_buf<grow_only>(message), recv_count(5)).extract();
        auto status = handle.wait(status_out());
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 0);
        EXPECT_EQ(status.source(), comm.root());
        EXPECT_EQ(status.tag(), 0);
        EXPECT_EQ(status.count<int>(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5, -1, -1, -1}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IrecvTest, recv_vector_with_explicit_size_grow_only_too_small) {
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
        auto handle = comm.irecv(recv_buf<grow_only>(message), recv_count(5)).extract();
        auto status = handle.wait(status_out());
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 0);
        EXPECT_EQ(status.source(), comm.root());
        EXPECT_EQ(status.tag(), 0);
        EXPECT_EQ(status.count<int>(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IrecvTest, recv_vector_with_input_status) {
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
        auto [handle, result] = comm.irecv(recv_buf<resize_to_fit>(message), recv_count_out()).extract();
        handle.wait(status_out(recv_status));
        EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result)>);
        EXPECT_EQ(recv_status.source(), comm.root());
        EXPECT_EQ(recv_status.tag(), 0);
        EXPECT_EQ(recv_status.count<int>(), 5);
        EXPECT_EQ(result.extract_recv_count(), 5);
        EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IrecvTest, recv_default_custom_container_without_recv_buf) {
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
        auto                         handle  = comm.irecv<int>();
        ::testing::OwnContainer<int> message = handle.wait();
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 1);
        EXPECT_EQ(message, ::testing::OwnContainer<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IrecvTest, recv_default_custom_container_without_recv_buf_but_with_recv_count) {
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
        auto handle = comm.irecv<int>(recv_count_out());
        auto result = handle.wait();
        EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result)>);
        ::testing::OwnContainer<int> message = result.extract_recv_buffer();
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 1);
        EXPECT_EQ(result.extract_recv_count(), 5);
        EXPECT_EQ(message, ::testing::OwnContainer<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IrecvTest, recv_default_custom_container_without_recv_buf_but_with_recv_count_recv_into_struct_binding) {
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
        auto handle                = comm.irecv<int>(recv_count_out());
        auto [message, recv_count] = handle.wait();
        // we should not probe for the message size inside of KaMPIng if we specify the recv count explicitly
        EXPECT_EQ(probe_counter, 1);
        EXPECT_EQ(recv_count, 5);
        EXPECT_EQ(message, ::testing::OwnContainer<int>({1, 2, 3, 4, 5}));
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IrecvTest, recv_from_proc_null) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    auto [handle, result] = comm.irecv(source(rank::null), recv_buf(v), recv_count_out()).extract();
    EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result)>);
    auto   status     = handle.wait(status_out());
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
TEST_F(IrecvTest, recv_from_invalid_tag) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    EXPECT_KASSERT_FAILS({ comm.irecv(recv_buf(v), tag(-1)); }, "invalid tag");
}
#endif

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST_F(IrecvTest, recv_from_invalid_tag_with_explicit_recv_count) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    EXPECT_KASSERT_FAILS({ comm.irecv(recv_buf(v), tag(-1), recv_count(1)); }, "invalid tag");
}
#endif

TEST_F(IrecvTest, recv_type_is_out_param) {
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
            auto [handle, result] =
                comm.irecv(recv_buf<resize_to_fit>(message), recv_type_out(recv_type), recv_count_out()).extract();
            auto status = handle.wait(status_out());
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

TEST_F(IrecvTest, non_trivial_recv_type) {
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
        MPI_Datatype int_padding_padding = ::testing::MPI_INT_padding_padding();
        MPI_Type_commit(&int_padding_padding);
        for (size_t other = 0; other < comm.size(); other++) {
            int const        default_init = -1;
            std::vector<int> message(3 * other, default_init);
            auto             handle =
                comm.irecv(recv_buf<no_resize>(message), source(other), recv_type(int_padding_padding)).extract();
            auto status = handle.wait(status_out());
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

TEST_F(IrecvTest, recv_buf_passthrough) {
    using namespace ::testing;
    Communicator comm;
    if (comm.size() < 2) {
        return;
    }
    if (comm.rank() == 0) {
        std::vector<int> v{{42, 1, 7, 5}};
        MPI_Send(v.data(), 4, MPI_INT, 1, 0, comm.mpi_communicator());
    } else if (comm.rank() == 1) {
        std::vector<int> buf(4);
        auto             req    = comm.irecv<int>(recv_buf(std::move(buf)), recv_count(4));
        auto             result = req.wait();
        EXPECT_THAT(result, ElementsAre(42, 1, 7, 5));
    }
    comm.barrier();
}

TEST_F(IrecvTest, recv_buf_passthrough_single_element) {
    using namespace ::testing;
    Communicator comm;
    if (comm.size() < 2) {
        return;
    }
    if (comm.rank() == 0) {
        int value = 43;
        MPI_Send(&value, 1, MPI_INT, 1, 0, comm.mpi_communicator());
    } else if (comm.rank() == 1) {
        int  value = 27;
        auto req   = comm.irecv(recv_count(1), recv_buf_out(std::move(value)));
        value      = req.wait();
        EXPECT_EQ(value, 43);
    }
    comm.barrier();
}
