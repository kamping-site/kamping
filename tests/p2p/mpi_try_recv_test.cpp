// This file is part of KaMPIng.
//
// Copyright 2023-2024 The KaMPIng Authors
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
#include "kamping/p2p/try_recv.hpp"

using namespace kamping;
using namespace ::testing;

KAMPING_MAKE_HAS_MEMBER(extract_status)
KAMPING_MAKE_HAS_MEMBER(extract_recv_buffer)

class TryRecvTest : public ::testing::Test {
    void SetUp() override {
        // This makes sure that messages don't spill from other tests.
        MPI_Barrier(MPI_COMM_WORLD);
    }
    void TearDown() override {
        // This makes sure that messages don't spill to other tests.
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

TEST_F(TryRecvTest, try_recv_vector_from_arbitrary_source) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;

    // No messages have been sent yet, so the try_recv() should return std::nullopt
    EXPECT_FALSE(comm.try_recv<int>().has_value());
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
        for (size_t other = 0; other < comm.size(); other++) {
            while (true) {
                auto result_opt = comm.try_recv<int>(status_out());
                // The messages might not yet be delivered.
                if (result_opt.has_value()) {
                    auto& result = result_opt.value();

                    EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
                    auto const status  = result.extract_status();
                    auto const source  = status.source();
                    auto       message = result.extract_recv_buffer();

                    EXPECT_EQ(status.tag(), source);
                    EXPECT_EQ(status.count<int>(), source);
                    EXPECT_EQ(message.size(), source);
                    EXPECT_EQ(message, std::vector(source, 42));

                    break;
                }
            }
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    EXPECT_EQ(comm.try_recv<long>(), std::nullopt);
}

TEST_F(TryRecvTest, try_recv_vector_from_explicit_source) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;

    // No messages have been sent yet, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
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
        for (size_t other = 0; other < comm.size(); other++) {
            while (true) {
                auto result_opt = comm.try_recv<int>(source(other), status_out());
                if (result_opt.has_value()) {
                    auto& result = result_opt.value();
                    EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
                    auto status  = result.extract_status();
                    auto source  = status.source();
                    auto message = result.extract_recv_buffer();
                    EXPECT_EQ(source, other);
                    EXPECT_EQ(status.tag(), source);
                    EXPECT_EQ(status.count<int>(), source);
                    EXPECT_EQ(message.size(), source);
                    EXPECT_EQ(message, std::vector(source, 42));
                    break;
                }
            }
        }
    }

    // Ensure that we have received all inflight messages.
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
}

TEST_F(TryRecvTest, try_recv_vector_from_explicit_source_and_explicit_tag) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;

    // No messages have been sent yet, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
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
        for (size_t other = 0; other < comm.size(); other++) {
            while (true) {
                auto result_opt = comm.try_recv<int>(source(other), tag(asserting_cast<int>(other)), status_out());
                if (result_opt.has_value()) {
                    auto& result = result_opt.value();
                    EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
                    auto status  = result.extract_status();
                    auto source  = status.source();
                    auto message = result.extract_recv_buffer();
                    EXPECT_EQ(source, other);
                    EXPECT_EQ(status.tag(), source);
                    EXPECT_EQ(status.count<int>(), source);
                    EXPECT_EQ(message.size(), source);
                    EXPECT_EQ(message, std::vector(source, 42));
                    break;
                }
            }
        }
    }

    // Ensure that we have received all inflight messages.
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
}

TEST_F(TryRecvTest, try_recv_vector_no_resize) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    MPI_Request  req = MPI_REQUEST_NULL;

    // No messages have been sent yet, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
    comm.barrier();

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
        std::vector<int> message(42, std::numeric_limits<int>::max());

        while (true) {
            auto result_opt = comm.try_recv(recv_buf<no_resize>(message), status_out());
            if (result_opt.has_value()) {
                auto& result = result_opt.value();
                EXPECT_TRUE(has_member_extract_status_v<decltype(result)>);
                auto status = result.extract_status();
                EXPECT_EQ(status.source(), comm.root());
                EXPECT_EQ(status.count<int>(), 5);
                EXPECT_EQ(status.tag(), 0);
                EXPECT_EQ(message.size(), 42);
                EXPECT_THAT(Span(message.data(), 5), ElementsAre(1, 2, 3, 4, 5));
                EXPECT_THAT(Span(message.data() + 5, 42 - 5), Each(std::numeric_limits<int>::max()));
                break;
            }
        }
    }

    MPI_Wait(&req, MPI_STATUS_IGNORE);
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
}

TEST_F(TryRecvTest, try_recv_vector_with_status_out) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    MPI_Request  req = MPI_REQUEST_NULL;

    // No messages have been sent yet, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
    comm.barrier();

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
        while (true) {
            bool const result_opt = comm.try_recv(recv_buf<resize_to_fit>(message), status_out(recv_status));
            if (result_opt) {
                EXPECT_EQ(recv_status.source(), comm.root());
                EXPECT_EQ(recv_status.tag(), 0);
                EXPECT_EQ(recv_status.count<int>(), 5);
                EXPECT_EQ(message, std::vector<int>({1, 2, 3, 4, 5}));
                break;
            }
        }
    }

    MPI_Wait(&req, MPI_STATUS_IGNORE);
    // No more messages are inflight, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
}

TEST_F(TryRecvTest, try_recv_default_custom_container_without_recv_buf) {
    Communicator<::testing::OwnContainer> comm;
    std::vector                           v{1, 2, 3, 4, 5};
    MPI_Request                           req = MPI_REQUEST_NULL;

    // No messages have been sent yet, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
    comm.barrier();

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
        while (true) {
            auto result_opt = comm.try_recv<int>();
            if (result_opt.has_value()) {
                auto&                        result  = result_opt.value();
                ::testing::OwnContainer<int> message = result;
                EXPECT_EQ(message, ::testing::OwnContainer<int>({1, 2, 3, 4, 5}));
                break;
            }
        }
    }

    MPI_Wait(&req, MPI_STATUS_IGNORE);
    // No more messages are inflight, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
}

TEST_F(TryRecvTest, try_recv_from_proc_null) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};

    // No messages have been sent yet, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
    comm.barrier();

    while (true) {
        auto result_opt = comm.try_recv(source(rank::null), recv_buf(v), status_out());
        if (result_opt.has_value()) {
            auto& result = result_opt.value();
            auto  status = result.extract_status();
            // recv did not touch the buffer
            EXPECT_EQ(v.size(), 5);
            EXPECT_EQ(v, std::vector({1, 2, 3, 4, 5}));
            EXPECT_EQ(status.source_signed(), MPI_PROC_NULL);
            EXPECT_EQ(status.tag(), MPI_ANY_TAG);
            break;
        }
    }

    // No messages are inflight, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
}

TEST_F(TryRecvTest, recv_type_is_out_param) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);

    // No messages have been sent yet, so the try_recv() should return std::nullopt
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
    comm.barrier();

    MPI_Request req;
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
            while (true) {
                std::vector<int> message;
                auto             result = comm.try_recv(
                    recv_buf<kamping::BufferResizePolicy::resize_to_fit>(message),
                    status_out(),
                    recv_type_out(recv_type)
                );
                if (result) {
                    auto status = result->extract_status();
                    auto source = status.source();
                    EXPECT_EQ(status.tag(), source);
                    EXPECT_EQ(status.count<int>(), source);
                    EXPECT_EQ(recv_type, MPI_INT);
                    EXPECT_EQ(message.size(), source);
                    EXPECT_EQ(message, std::vector(source, 42));
                    break;
                }
            }
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    EXPECT_EQ(comm.try_recv<int>(), std::nullopt);
}

TEST_F(TryRecvTest, non_trivial_recv_type) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);

    int const        default_init = -1;
    std::vector<int> message;

    EXPECT_FALSE(comm.try_recv(recv_buf<no_resize>(message), recv_type(MPI_INT_padding_padding())));
    comm.barrier();

    MPI_Request req;
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
            message.resize(3 * other, default_init);
            while (true) {
                auto result = comm.try_recv(
                    recv_buf<no_resize>(message),
                    status_out(),
                    source(other),
                    recv_type(int_padding_padding)
                );
                if (result) {
                    auto status = result->extract_status();
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
                    break;
                }
            }
        }
        MPI_Type_free(&int_padding_padding);
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    EXPECT_FALSE(comm.try_recv(recv_buf<no_resize>(message), recv_type(MPI_INT_padding_padding())));
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST_F(TryRecvTest, try_recv_from_invalid_tag) {
    Communicator comm;
    std::vector  v{1, 2, 3, 4, 5};
    EXPECT_KASSERT_FAILS({ comm.try_recv(recv_buf(v), status_out(), tag(-1)); }, "invalid tag");
}
#endif
