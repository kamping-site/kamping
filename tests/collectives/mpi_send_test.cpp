// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/collectives/send.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;

TEST(SendTest, send_vector) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), receiver(other_rank));
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(4);
        MPI_Status       status;
        MPI_Recv(
            msg.data(),
            static_cast<int>(msg.size()),
            MPI_INT,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            comm.mpi_communicator(),
            &status
        );
        ASSERT_EQ(msg, (std::vector<int>{42, 3, 8, 7}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    }
}

TEST(SendTest, send_vector_with_tag) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), receiver(other_rank), tag(42));
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(4);
        MPI_Status       status;
        MPI_Recv(
            msg.data(),
            static_cast<int>(msg.size()),
            MPI_INT,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            comm.mpi_communicator(),
            &status
        );
        ASSERT_EQ(msg, (std::vector<int>{42, 3, 8, 7}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 42);
    }
}

TEST(SendTest, send_vector_with_enum_tag_recv_out_of_order) {
    enum class Tag {
        control_message = 13,
        data_message    = 27,
    };
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        comm.send(send_buf(std::vector<int>{}), receiver(other_rank), tag(Tag::control_message));

        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), receiver(other_rank), tag(Tag::data_message));
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg;
        MPI_Status       status;
        MPI_Recv(
            msg.data(),
            0,
            MPI_INT,
            MPI_ANY_SOURCE,
            static_cast<int>(Tag::control_message),
            comm.mpi_communicator(),
            &status
        );
        ASSERT_EQ(msg.size(), 0);
        ASSERT_EQ(status.MPI_TAG, static_cast<int>(Tag::control_message));

        msg.resize(4);
        MPI_Recv(
            msg.data(),
            static_cast<int>(msg.size()),
            MPI_INT,
            MPI_ANY_SOURCE,
            static_cast<int>(Tag::data_message),
            comm.mpi_communicator(),
            &status
        );
        ASSERT_EQ(msg, (std::vector<int>{42, 3, 8, 7}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, static_cast<int>(Tag::data_message));
    }
}
