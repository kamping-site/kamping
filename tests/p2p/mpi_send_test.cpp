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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/parameter_objects.hpp"

using namespace ::kamping;
using namespace ::testing;

// Note: These invariants tested here only hold when the tests are executed using more than one MPI rank!

static size_t call_hierarchy_level = 0;
static size_t send_counter         = 0;
static size_t bsend_counter        = 0;
static size_t ssend_counter        = 0;
static size_t rsend_counter        = 0;

class SendTest : public ::testing::Test {
    void SetUp() override {
        Communicator comm;
        ASSERT_GT(comm.size(), 1)
            << "The invariants tested here only hold when the tests are executed using more than one MPI rank!";
        call_hierarchy_level = 0;
        send_counter         = 0;
        bsend_counter        = 0;
        ssend_counter        = 0;
        rsend_counter        = 0;
    }
    void TearDown() override {
        call_hierarchy_level = 0;
        send_counter         = 0;
        bsend_counter        = 0;
        ssend_counter        = 0;
        rsend_counter        = 0;
    }
};

int MPI_Send(void const* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    call_hierarchy_level++;
    auto errcode = PMPI_Send(buf, count, datatype, dest, tag, comm);
    // a send call may call another operation in its implementation
    // this ensure that we only count the top level send operation
    if (call_hierarchy_level == 1) {
        send_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

int MPI_Bsend(void const* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    call_hierarchy_level++;
    auto errcode = PMPI_Bsend(buf, count, datatype, dest, tag, comm);
    if (call_hierarchy_level == 1) {
        bsend_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

int MPI_Ssend(void const* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    call_hierarchy_level++;
    auto errcode = PMPI_Ssend(buf, count, datatype, dest, tag, comm);
    if (call_hierarchy_level == 1) {
        ssend_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

int MPI_Rsend(void const* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    call_hierarchy_level++;
    auto errcode = PMPI_Rsend(buf, count, datatype, dest, tag, comm);
    if (call_hierarchy_level == 1) {
        rsend_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

TEST_F(SendTest, send_vector) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), destination(other_rank));
        ASSERT_EQ(send_counter, 1);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 0);
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

TEST_F(SendTest, send_vector_with_explicit_send_count) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), send_count(2), destination(other_rank));
        EXPECT_EQ(send_counter, 1);
        EXPECT_EQ(bsend_counter, 0);
        EXPECT_EQ(ssend_counter, 0);
        EXPECT_EQ(rsend_counter, 0);
    } else if (comm.rank() == other_rank) {
        std::vector<int> result(4, -1);
        MPI_Status       status;
        MPI_Message      msg;
        MPI_Mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm.mpi_communicator(), &msg, &status);
        int count;
        MPI_Get_count(&status, MPI_INT, &count);
        EXPECT_EQ(count, 2);
        MPI_Mrecv(result.data(), count, MPI_INT, &msg, MPI_STATUS_IGNORE);
        EXPECT_THAT(result, ElementsAre(42, 3, -1, -1));
        EXPECT_EQ(status.MPI_SOURCE, comm.root());
        EXPECT_EQ(status.MPI_TAG, 0);
    }
}

TEST_F(SendTest, send_vector_null) {
    Communicator     comm;
    std::vector<int> values{42, 3, 8, 7};
    comm.send(send_buf(values), destination(rank::null));
    ASSERT_EQ(send_counter, 1);
    ASSERT_EQ(bsend_counter, 0);
    ASSERT_EQ(ssend_counter, 0);
    ASSERT_EQ(rsend_counter, 0);
}

TEST_F(SendTest, send_vector_with_tag) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), destination(other_rank), tag(42));
        ASSERT_EQ(send_counter, 1);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 0);
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

TEST_F(SendTest, send_vector_with_enum_tag_recv_out_of_order) {
    enum class Tag {
        control_message = 13,
        data_message    = 27,
    };
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        comm.send(send_buf(std::vector<int>{}), destination(other_rank), tag(Tag::control_message));
        ASSERT_EQ(send_counter, 1);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 0);

        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), destination(other_rank), tag(Tag::data_message));
        ASSERT_EQ(send_counter, 2);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 0);
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

TEST_F(SendTest, send_vector_standard) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), destination(other_rank), send_mode(send_modes::standard));
        ASSERT_EQ(send_counter, 1);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 0);
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

TEST_F(SendTest, send_vector_buffered) {
    Communicator comm;

    // allocate the minimum required buffer size and attach it
    int pack_size;
    MPI_Pack_size(4, MPI_INT, MPI_COMM_WORLD, &pack_size);
    auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(pack_size + MPI_BSEND_OVERHEAD));
    MPI_Buffer_attach(buffer.get(), pack_size + MPI_BSEND_OVERHEAD);

    auto other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), destination(other_rank), send_mode(send_modes::buffered));
        ASSERT_EQ(send_counter, 0);
        ASSERT_EQ(bsend_counter, 1);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 0);
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

    // detach the buffer
    void* dummy;
    int   dummy_size;
    MPI_Buffer_detach(&dummy, &dummy_size);
}

TEST_F(SendTest, send_vector_synchronous) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), destination(other_rank), send_mode(send_modes::synchronous));
        ASSERT_EQ(send_counter, 0);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 1);
        ASSERT_EQ(rsend_counter, 0);
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

TEST_F(SendTest, send_vector_ready) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        // ensure that the receive is posted before the send is started
        MPI_Barrier(comm.mpi_communicator());
        std::vector<int> values{42, 3, 8, 7};
        comm.send(send_buf(values), destination(other_rank), send_mode(send_modes::ready));
        ASSERT_EQ(send_counter, 0);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 1);
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(4);
        MPI_Status       status;
        MPI_Request      req;
        // ensure that the receive is posted before the send is started
        MPI_Irecv(
            msg.data(),
            static_cast<int>(msg.size()),
            MPI_INT,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            comm.mpi_communicator(),
            &req
        );
        MPI_Barrier(comm.mpi_communicator());
        MPI_Wait(&req, &status);
        ASSERT_EQ(msg, (std::vector<int>{42, 3, 8, 7}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    } else {
        // ensure that the receive is posted before the send is started
        MPI_Barrier(comm.mpi_communicator());
    }
}

TEST_F(SendTest, send_vector_bsend) {
    Communicator comm;

    // allocate the minimum required buffer size and attach it
    int pack_size;
    MPI_Pack_size(4, MPI_INT, MPI_COMM_WORLD, &pack_size);
    auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(pack_size + MPI_BSEND_OVERHEAD));
    MPI_Buffer_attach(buffer.get(), pack_size + MPI_BSEND_OVERHEAD);

    auto other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.bsend(send_buf(values), destination(other_rank));
        ASSERT_EQ(send_counter, 0);
        ASSERT_EQ(bsend_counter, 1);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 0);
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

    // detach the buffer
    void* dummy;
    int   dummy_size;
    MPI_Buffer_detach(&dummy, &dummy_size);
}

TEST_F(SendTest, send_vector_ssend) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        comm.ssend(send_buf(values), destination(other_rank));
        ASSERT_EQ(send_counter, 0);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 1);
        ASSERT_EQ(rsend_counter, 0);
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

TEST_F(SendTest, send_vector_rsend) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        // ensure that the receive is posted before the send is started
        MPI_Barrier(comm.mpi_communicator());
        std::vector<int> values{42, 3, 8, 7};
        comm.rsend(send_buf(values), destination(other_rank));
        ASSERT_EQ(send_counter, 0);
        ASSERT_EQ(bsend_counter, 0);
        ASSERT_EQ(ssend_counter, 0);
        ASSERT_EQ(rsend_counter, 1);
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(4);
        MPI_Status       status;
        MPI_Request      req;
        // ensure that the receive is posted before the send is started
        MPI_Irecv(
            msg.data(),
            static_cast<int>(msg.size()),
            MPI_INT,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            comm.mpi_communicator(),
            &req
        );
        MPI_Barrier(comm.mpi_communicator());
        MPI_Wait(&req, &status);
        ASSERT_EQ(msg, (std::vector<int>{42, 3, 8, 7}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    } else {
        // ensure that the receive is posted before the send is started
        MPI_Barrier(comm.mpi_communicator());
    }
}

TEST_F(SendTest, non_trivial_send_type_send) {
    // a sender rank sends its rank two times with padding to a receiver rank; this rank receives the ranks without
    // padding.
    Communicator comm;
    auto         other_rank          = (comm.root() + 1) % comm.size();
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();
    MPI_Type_commit(&int_padding_padding);
    if (comm.is_root()) {
        std::vector<int> values{comm.rank_signed(), -1, -1, comm.rank_signed(), -1, -1};
        comm.send(send_buf(values), send_type(int_padding_padding), send_count(2), destination(other_rank));
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(2);
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
        ASSERT_EQ(msg, (std::vector<int>{comm.root_signed(), comm.root_signed()}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    }
    MPI_Type_free(&int_padding_padding);
}

TEST_F(SendTest, non_trivial_send_type_ssend) {
    // a sender rank sends its rank two times with padding to a receiver rank; this rank receives the ranks without
    // padding.
    Communicator comm;
    auto         other_rank          = (comm.root() + 1) % comm.size();
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();
    MPI_Type_commit(&int_padding_padding);
    if (comm.is_root()) {
        std::vector<int> values{comm.rank_signed(), -1, -1, comm.rank_signed(), -1, -1};
        comm.ssend(send_buf(values), send_type(int_padding_padding), send_count(2), destination(other_rank));
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(2);
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
        ASSERT_EQ(msg, (std::vector<int>{comm.root_signed(), comm.root_signed()}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    }
    MPI_Type_free(&int_padding_padding);
}

TEST_F(SendTest, non_trivial_send_type_bsend) {
    // a sender rank sends its rank two times with padding to a receiver rank; this rank receives the ranks without
    // padding.
    Communicator comm;
    auto         other_rank          = (comm.root() + 1) % comm.size();
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();
    MPI_Type_commit(&int_padding_padding);

    // allocate the minimum required buffer size and attach it
    int pack_size;
    MPI_Pack_size(2, int_padding_padding, MPI_COMM_WORLD, &pack_size);
    auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(pack_size + MPI_BSEND_OVERHEAD));
    MPI_Buffer_attach(buffer.get(), pack_size + MPI_BSEND_OVERHEAD);

    if (comm.is_root()) {
        std::vector<int> values{comm.rank_signed(), -1, -1, comm.rank_signed(), -1, -1};
        comm.bsend(send_buf(values), send_type(int_padding_padding), send_count(2), destination(other_rank));
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(2);
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
        ASSERT_EQ(msg, (std::vector<int>{comm.root_signed(), comm.root_signed()}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    }
    MPI_Type_free(&int_padding_padding);

    // detach the buffer
    void* dummy;
    int   dummy_size;
    MPI_Buffer_detach(&dummy, &dummy_size);
}

TEST_F(SendTest, non_trivial_send_type_rsend) {
    // a sender rank sends its rank two times with padding to a receiver rank; this rank receives the ranks without
    // padding.
    Communicator comm;
    auto         other_rank          = (comm.root() + 1) % comm.size();
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();
    MPI_Type_commit(&int_padding_padding);
    if (comm.is_root()) {
        // ensure that the receive is posted before the send is started
        MPI_Barrier(comm.mpi_communicator());
        std::vector<int> values{comm.rank_signed(), -1, -1, comm.rank_signed(), -1, -1};
        comm.rsend(send_buf(values), send_type(int_padding_padding), send_count(2), destination(other_rank));
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(2);
        MPI_Status       status;
        MPI_Request      req;
        // ensure that the receive is posted before the send is started
        MPI_Irecv(
            msg.data(),
            static_cast<int>(msg.size()),
            MPI_INT,
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            comm.mpi_communicator(),
            &req
        );
        MPI_Barrier(comm.mpi_communicator());
        MPI_Wait(&req, &status);
        ASSERT_EQ(msg, (std::vector<int>{comm.root_signed(), comm.root_signed()}));
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    } else {
        // all other ranks also have to participate in the barrier
        MPI_Barrier(comm.mpi_communicator());
    }
    MPI_Type_free(&int_padding_padding);
}
