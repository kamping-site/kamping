// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "../test_assertions.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/sendrecv.hpp"
#include "kamping/parameter_objects.hpp"

using namespace kamping;
using namespace ::testing;

TEST(SendrecvTest, sendrecv_vector_cyclic) {
    Communicator comm;

    std::vector<int> input(1, comm.rank_signed());
    std::vector<int> message;
    auto             sent_to   = comm.rank_shifted_cyclic(1);
    auto             sent_from = comm.rank_shifted_cyclic(-1);

    comm.sendrecv(
        send_buf(input),
        send_count(1),
        destination(sent_to),
        recv_buf<BufferResizePolicy::resize_to_fit>(message),
        recv_count(1)
    );

    ASSERT_EQ(message, std::vector<int>{static_cast<int>(sent_from)});
    ASSERT_EQ(message.size(), 1);
}

TEST(SendrecvTest, sendrecv_vector_cyclic_wo_recv_buf) {
    Communicator comm;

    std::vector<int> input(1, comm.rank_signed());
    auto             sent_to   = comm.rank_shifted_cyclic(1);
    auto             sent_from = comm.rank_shifted_cyclic(-1);

    auto message = comm.sendrecv<int>(send_buf(input), send_count(1), destination(sent_to), recv_count(1));

    ASSERT_EQ(message, std::vector<int>{static_cast<int>(sent_from)});
    ASSERT_EQ(message.size(), 1);
}

TEST(SendrecvTest, sendrecv_vector_cyclic_wo_recv_count) {
    Communicator comm;

    std::vector<int> input(42, comm.rank_signed());
    auto             sent_to   = comm.rank_shifted_cyclic(1);
    auto             sent_from = comm.rank_shifted_cyclic(-1);

    auto message = comm.sendrecv<int>(send_buf(input), send_count(42), destination(sent_to));

    ASSERT_EQ(message, std::vector<int>(42, static_cast<int>(sent_from)));
    ASSERT_EQ(message.size(), 42);
}

TEST(SendrecvTest, send_and_recv_with_sendrecv) {
    Communicator comm;
    ASSERT_GT(comm.size(), 1)
        << "The invariants tested here only hold when the tests are executed using more than one MPI rank!";
    auto other_rank = (comm.root() + 1) % comm.size();

    if (comm.is_root()) {
        std::vector<int> root_recv(3, 0);
        MPI_Status       status;
        MPI_Recv(
            &root_recv[0],
            3,
            MPI_INT,
            static_cast<int>(other_rank),
            MPI_ANY_TAG,
            comm.mpi_communicator(),
            &status
        );
        ASSERT_EQ(root_recv, std::vector({11, 12, 13}));

        std::vector<int> root_send{4, 5, 6, 7, 8, 9};
        MPI_Send(
            &root_send[0],
            static_cast<int>(root_send.size()),
            MPI_INT,
            static_cast<int>(other_rank),
            comm.rank_signed(),
            comm.mpi_communicator()
        );
    }

    if (comm.rank_shifted_cyclic(-1) == comm.root()) {
        std::vector<int> msg{11, 12, 13};
        auto             message = comm.sendrecv<int>(send_buf(msg), destination(comm.root()), recv_count(6));

        ASSERT_EQ(message, std::vector({4, 5, 6, 7, 8, 9}));
    }
}

TEST(SendrecvTest, sendrecv_cyclic_all_params) {
    Communicator comm;

    std::vector<int> input(1, comm.rank_signed());
    std::vector<int> message(1);
    auto             sent_to   = comm.rank_shifted_cyclic(1);
    auto             sent_from = comm.rank_shifted_cyclic(-1);

    comm.sendrecv(
        send_buf(input),
        send_count(1),
        destination(sent_to),
        recv_buf<BufferResizePolicy::no_resize>(message),
        recv_count(1),
        send_type(MPI_INT),
        send_tag(7),
        recv_type(MPI_INT),
        source(rank::any),
        recv_tag(tags::any),
        status(ignore<>)
    );

    ASSERT_EQ(message, std::vector<int>{static_cast<int>(sent_from)});
    ASSERT_EQ(message.size(), 1);
}

TEST(SendrecvTest, sendrecv_cyclic_only_req_params) {
    Communicator comm;

    std::vector<int> input(42, comm.rank_signed());
    auto             sent_to   = comm.rank_shifted_cyclic(1);
    auto             sent_from = comm.rank_shifted_cyclic(-1);

    auto message = comm.sendrecv<int>(send_buf(input), destination(sent_to));

    ASSERT_EQ(message, std::vector<int>(42, static_cast<int>(sent_from)));
    ASSERT_EQ(message.size(), 42);
}

TEST(SendrecvTest, sendrecv_cyclic_with_status) {
    Communicator comm;

    std::vector<int> input(42, comm.rank_signed());
    auto             sent_to   = comm.rank_shifted_cyclic(1);
    auto             sent_from = comm.rank_shifted_cyclic(-1);

    auto result = comm.sendrecv<int>(send_buf(input), destination(sent_to), status_out(), recv_type_out());

    auto message = result.extract_recv_buf();
    ASSERT_EQ(message, std::vector<int>(42, static_cast<int>(sent_from)));
    ASSERT_EQ(message.size(), 42);
    auto status = result.extract_status();
    auto source = status.source();
    ASSERT_EQ(source, sent_from);
    EXPECT_EQ(result.extract_recv_type(), MPI_INT);
}

TEST(SendrecvTest, sendrecv_different_send_and_recv_count) {
    Communicator comm;

    std::vector<int> input(static_cast<size_t>(comm.rank_signed() + 10), comm.rank_signed());
    auto             sent_to   = comm.rank_shifted_cyclic(1);
    auto             sent_from = comm.rank_shifted_cyclic(-1);

    auto message = comm.sendrecv<int>(send_buf(input), destination(sent_to));

    ASSERT_EQ(message, std::vector<int>(sent_from + 10, static_cast<int>(sent_from)));
    ASSERT_EQ(message.size(), sent_from + 10);
}

TEST(SendrecvTest, sendrecv_custom_container) {
    Communicator      comm;
    OwnContainer<int> values(4);
    for (int& value: values) {
        value = comm.rank_signed();
    }
    auto sent_to   = comm.rank_shifted_cyclic(1);
    auto sent_from = comm.rank_shifted_cyclic(-1);

    auto message = comm.sendrecv<int>(send_buf(values), destination(sent_to));

    ASSERT_EQ(message.size(), 4);
    for (int i: message) {
        EXPECT_EQ(i, sent_from);
    }
}

TEST(SendrecvTest, sendrecv_with_MPI_sendrecv) {
    Communicator comm;
    ASSERT_GT(comm.size(), 1)
        << "The invariants tested here only hold when the tests are executed using more than one MPI rank!";
    auto other_rank = (comm.root() + 1) % comm.size();

    if (comm.is_root()) {
        std::vector<int> root_send(3, 0);
        std::vector<int> root_recv(3);
        MPI_Status       status;
        MPI_Sendrecv(
            &root_send[0],
            3,
            MPI_INT,
            static_cast<int>(other_rank),
            5,
            &root_recv[0],
            3,
            MPI_INT,
            asserting_cast<int>(other_rank),
            MPI_ANY_TAG,
            comm.mpi_communicator(),
            &status
        );
        ASSERT_EQ(root_recv, std::vector<int>({11, 12, 13}));
    }

    if (comm.rank_shifted_cyclic(-1) == comm.root()) {
        std::vector<int> msg{11, 12, 13};
        auto             message = comm.sendrecv<int>(send_buf(msg), destination(comm.root()), recv_count(3));

        ASSERT_EQ(message, std::vector<int>({0, 0, 0}));
    }
}

TEST(SendrecvTest, sendrecv_different_types) {
    Communicator comm;
    ASSERT_GT(comm.size(), 1)
        << "The invariants tested here only hold when the tests are executed using more than one MPI rank!";
    auto other_rank = (comm.root() + 1) % comm.size();

    if (comm.is_root()) {
        std::vector<char> root_send{'a', 'b', 'c'};
        auto              message = comm.sendrecv<int>(send_buf(root_send), destination(other_rank));
        ASSERT_EQ(message, std::vector<int>({11, 12, 13, 14}));
    }

    if (comm.rank_shifted_cyclic(-1) == comm.root()) {
        std::vector<int> msg{11, 12, 13, 14};
        auto             message = comm.sendrecv<char>(send_buf(msg), destination(comm.root()));

        ASSERT_EQ(message, std::vector<char>({'a', 'b', 'c'}));
    }
}

TEST(SendrecvTest, sendrecv_different_types_with_explicit_buffer) {
    Communicator comm;
    ASSERT_GT(comm.size(), 1)
        << "The invariants tested here only hold when the tests are executed using more than one MPI rank!";
    auto other_rank = (comm.root() + 1) % comm.size();

    if (comm.is_root()) {
        std::vector<char> root_send{'a', 'b', 'c'};
        std::vector<int>  root_recv;
        comm.sendrecv(
            send_buf(root_send),
            destination(other_rank),
            recv_buf<BufferResizePolicy::resize_to_fit>(root_recv)
        );
        ASSERT_EQ(root_recv, std::vector<int>({11, 12, 13, 14}));
    }

    if (comm.rank_shifted_cyclic(-1) == comm.root()) {
        std::vector<int>  msg_send{11, 12, 13, 14};
        std::vector<char> msg_recv;
        comm.sendrecv(
            send_buf(msg_send),
            destination(comm.root()),
            recv_buf<BufferResizePolicy::resize_to_fit>(msg_recv)
        );

        ASSERT_EQ(msg_recv, std::vector<char>({'a', 'b', 'c'}));
    }
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
TEST(SendrecvTest, sendrecv_cyclic_with_explicit_size_no_resize_too_small) {
    Communicator comm;

    std::vector<int> input(5, comm.rank_signed());
    std::vector<int> msg_recv;
    auto             sent_to = comm.rank_shifted_cyclic(1);

    EXPECT_KASSERT_FAILS(
        {
            comm.sendrecv(
                send_buf(input),
                destination(sent_to),
                recv_buf<BufferResizePolicy::no_resize>(msg_recv),
                recv_count(5)
            );
        },
        "Recv buffer is not large enough to hold all received elements."
    );
}
#endif