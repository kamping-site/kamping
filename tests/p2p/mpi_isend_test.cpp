// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <set>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/parameter_objects.hpp"

using namespace ::kamping;
using namespace ::testing;

// Note: These invariants tested here only hold when the tests are executed using more than one MPI rank!

static size_t                call_hierarchy_level = 0;
static size_t                isend_counter        = 0;
static size_t                ibsend_counter       = 0;
static size_t                issend_counter       = 0;
static size_t                irsend_counter       = 0;
static std::set<MPI_Request> initialized_requests;
static std::set<MPI_Request> completed_requests;

class ISendTest : public ::testing::Test {
    void SetUp() override {
        Communicator comm;
        ASSERT_GT(comm.size(), 1)
            << "The invariants tested here only hold when the tests are executed using more than one MPI rank!";
        call_hierarchy_level = 0;
        isend_counter        = 0;
        ibsend_counter       = 0;
        issend_counter       = 0;
        irsend_counter       = 0;
        initialized_requests.clear();
        completed_requests.clear();
    }
    void TearDown() override {
        call_hierarchy_level = 0;
        isend_counter        = 0;
        ibsend_counter       = 0;
        issend_counter       = 0;
        irsend_counter       = 0;
        EXPECT_EQ(initialized_requests, completed_requests);
        initialized_requests.clear();
        completed_requests.clear();
    }
};

int MPI_Isend(
    void const* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request
) {
    call_hierarchy_level++;
    auto errcode = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    initialized_requests.insert(*request);
    // a send call may call another operation in its implementation
    // this ensure that we only count the top level send operation
    if (call_hierarchy_level == 1) {
        isend_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

int MPI_Ibsend(
    void const* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request
) {
    call_hierarchy_level++;
    auto errcode = PMPI_Ibsend(buf, count, datatype, dest, tag, comm, request);
    initialized_requests.insert(*request);
    if (call_hierarchy_level == 1) {
        ibsend_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

int MPI_Issend(
    void const* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request
) {
    call_hierarchy_level++;
    auto errcode = PMPI_Issend(buf, count, datatype, dest, tag, comm, request);
    initialized_requests.insert(*request);
    if (call_hierarchy_level == 1) {
        issend_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

int MPI_Irsend(
    void const* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request
) {
    call_hierarchy_level++;
    auto errcode = PMPI_Irsend(buf, count, datatype, dest, tag, comm, request);
    initialized_requests.insert(*request);
    if (call_hierarchy_level == 1) {
        irsend_counter++;
    }
    call_hierarchy_level--;
    return errcode;
}

int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request) {
    auto errcode = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    initialized_requests.insert(*request);
    return errcode;
}

int MPI_Test(MPI_Request* request, int* flag, MPI_Status* status) {
    MPI_Request in_request = *request;
    int         errcode    = PMPI_Test(request, flag, status);
    if (flag && in_request != MPI_REQUEST_NULL) {
        completed_requests.insert(in_request);
    }
    return errcode;
}

int MPI_Wait(MPI_Request* request, MPI_Status* status) {
    if (*request != MPI_REQUEST_NULL) {
        completed_requests.insert(*request);
    }
    return PMPI_Wait(request, status);
}

int MPI_Waitall(int count, MPI_Request* array_of_requests, MPI_Status* array_of_statuses) {
    for (int i = 0; i < count; i++) {
        if (*(array_of_requests + i) != MPI_REQUEST_NULL) {
            completed_requests.insert(*(array_of_requests + i));
        }
    }
    return PMPI_Waitall(count, array_of_requests, array_of_statuses);
}

// The following test are essentially the same as for blocking send with just awaiting the request. See below for
// additional tests.

TEST_F(ISendTest, send_vector) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.isend(send_buf(values), destination(other_rank));
        ASSERT_EQ(isend_counter, 1);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 0);
        req.wait();
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

TEST_F(ISendTest, send_vector_with_explicit_send_count) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.isend(send_buf(values), send_count(2), destination(other_rank));
        EXPECT_EQ(isend_counter, 1);
        EXPECT_EQ(ibsend_counter, 0);
        EXPECT_EQ(issend_counter, 0);
        EXPECT_EQ(irsend_counter, 0);
        req.wait();
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

TEST_F(ISendTest, send_vector_null) {
    Communicator     comm;
    std::vector<int> values{42, 3, 8, 7};
    auto             req = comm.isend(send_buf(values), destination(rank::null));
    ASSERT_EQ(isend_counter, 1);
    ASSERT_EQ(ibsend_counter, 0);
    ASSERT_EQ(issend_counter, 0);
    ASSERT_EQ(irsend_counter, 0);
    req.wait();
}

TEST_F(ISendTest, send_vector_with_tag) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.isend(send_buf(values), destination(other_rank), tag(42));
        ASSERT_EQ(isend_counter, 1);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 0);
        req.wait();
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

TEST_F(ISendTest, send_vector_with_enum_tag_recv_out_of_order) {
    enum class Tag {
        control_message = 13,
        data_message    = 27,
    };
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        auto req1 =
            comm.isend(send_buf(std::vector<int>{}), destination(other_rank), tag(Tag::control_message)).extract();
        ASSERT_EQ(isend_counter, 1);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 0);

        std::vector<int> values{42, 3, 8, 7};
        auto             req2 = comm.isend(send_buf(values), destination(other_rank), tag(Tag::data_message)).extract();
        ASSERT_EQ(isend_counter, 2);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 0);
        kamping::requests::wait_all(req1, req2);
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

TEST_F(ISendTest, send_vector_standard) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.isend(send_buf(values), destination(other_rank), send_mode(send_modes::standard));
        ASSERT_EQ(isend_counter, 1);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 0);
        req.wait();
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

TEST_F(ISendTest, send_vector_buffered) {
    Communicator comm;

    // allocate the minimum required buffer size and attach it
    int pack_size;
    MPI_Pack_size(4, MPI_INT, MPI_COMM_WORLD, &pack_size);
    auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(pack_size + MPI_BSEND_OVERHEAD));
    MPI_Buffer_attach(buffer.get(), pack_size + MPI_BSEND_OVERHEAD);

    auto other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.isend(send_buf(values), destination(other_rank), send_mode(send_modes::buffered));
        ASSERT_EQ(isend_counter, 0);
        ASSERT_EQ(ibsend_counter, 1);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 0);
        req.wait();
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

TEST_F(ISendTest, send_vector_synchronous) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        auto req = comm.isend(send_buf(values), destination(other_rank), send_mode(send_modes::synchronous));
        ASSERT_EQ(isend_counter, 0);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 1);
        ASSERT_EQ(irsend_counter, 0);
        req.wait();
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

TEST_F(ISendTest, send_vector_ready) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        // ensure that the receive is posted before the send is started
        MPI_Barrier(comm.mpi_communicator());
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.isend(send_buf(values), destination(other_rank), send_mode(send_modes::ready));
        ASSERT_EQ(isend_counter, 0);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 1);
        req.wait();
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

TEST_F(ISendTest, send_vector_bsend) {
    Communicator comm;

    // allocate the minimum required buffer size and attach it
    int pack_size;
    MPI_Pack_size(4, MPI_INT, MPI_COMM_WORLD, &pack_size);
    auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(pack_size + MPI_BSEND_OVERHEAD));
    MPI_Buffer_attach(buffer.get(), pack_size + MPI_BSEND_OVERHEAD);

    auto other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.ibsend(send_buf(values), destination(other_rank));
        ASSERT_EQ(isend_counter, 0);
        ASSERT_EQ(ibsend_counter, 1);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 0);
        req.wait();
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

TEST_F(ISendTest, send_vector_ssend) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.issend(send_buf(values), destination(other_rank));
        ASSERT_EQ(isend_counter, 0);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 1);
        ASSERT_EQ(irsend_counter, 0);
        req.wait();
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

TEST_F(ISendTest, send_vector_rsend) {
    Communicator comm;
    auto         other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        // ensure that the receive is posted before the send is started
        MPI_Barrier(comm.mpi_communicator());
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.irsend(send_buf(values), destination(other_rank));
        ASSERT_EQ(isend_counter, 0);
        ASSERT_EQ(ibsend_counter, 0);
        ASSERT_EQ(issend_counter, 0);
        ASSERT_EQ(irsend_counter, 1);
        req.wait();
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

// Here start the more interesting tests.
TEST_F(ISendTest, poor_mans_broadcast) {
    Communicator comm;
    int          result;
    if (comm.is_root()) {
        result = 42;
        std::vector<Request> requests(comm.size());
        for (std::size_t i = 0; i < comm.size(); i++) {
            if (i != comm.rank()) {
                comm.isend(send_buf(result), destination(i), request(requests[i]));
            }
        }
        kamping::requests::wait_all(requests);
    } else {
        result = comm.recv<int>(source(comm.root()))[0];
    }
    EXPECT_EQ(result, 42);
}

TEST_F(ISendTest, poor_mans_broadcast_with_test) {
    Communicator comm;
    int          result;
    if (comm.is_root()) {
        result = 42;
        std::vector<Request> requests(comm.size());
        for (std::size_t i = 0; i < comm.size(); i++) {
            if (i != comm.rank()) {
                comm.isend(send_buf(result), destination(i), request(requests[i]));
            }
        }
        bool any_test_failed;
        do {
            any_test_failed = false;
            for (auto& req: requests) {
                if (!req.test()) {
                    any_test_failed = true;
                }
            }
        } while (any_test_failed);
    } else {
        result = comm.recv<int>(source(comm.root()))[0];
    }
    EXPECT_EQ(result, 42);
}

TEST_F(ISendTest, non_trivial_send_type_isend) {
    // a sender rank sends its rank two times with padding to a receiver rank; this rank receives the ranks without
    // padding.
    Communicator comm;
    auto         other_rank          = (comm.root() + 1) % comm.size();
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();
    MPI_Type_commit(&int_padding_padding);
    if (comm.is_root()) {
        std::vector<int> values{comm.rank_signed(), -1, -1, comm.rank_signed(), -1, -1};
        auto req = comm.isend(send_buf(values), send_type(int_padding_padding), send_count(2), destination(other_rank));
        req.wait();
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

TEST_F(ISendTest, non_trivial_send_type_issend) {
    // a sender rank sends its rank two times with padding to a receiver rank; this rank receives the ranks without
    // padding.
    Communicator comm;
    auto         other_rank          = (comm.root() + 1) % comm.size();
    MPI_Datatype int_padding_padding = MPI_INT_padding_padding();
    MPI_Type_commit(&int_padding_padding);
    if (comm.is_root()) {
        std::vector<int> values{comm.rank_signed(), -1, -1, comm.rank_signed(), -1, -1};
        auto             req =
            comm.issend(send_buf(values), send_type(int_padding_padding), send_count(2), destination(other_rank));
        req.wait();
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

TEST_F(ISendTest, non_trivial_send_type_ibsend) {
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
        auto             req =
            comm.ibsend(send_buf(values), send_type(int_padding_padding), send_count(2), destination(other_rank));
        req.wait();
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

TEST_F(ISendTest, non_trivial_send_type_irsend) {
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
        auto             req =
            comm.irsend(send_buf(values), send_type(int_padding_padding), send_count(2), destination(other_rank));
        req.wait();
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

TEST_F(ISendTest, send_buf_ownership) {
    Communicator comm;
    if (comm.size() < 2) {
        return;
    }
    if (comm.rank() == 1) {
        std::vector<int> values{42, 3, 8, 7};
        auto             req = comm.isend(send_buf_out(std::move(values)), destination(0));
        EXPECT_EQ(values.size(), 0);
        values = req.wait();
        EXPECT_THAT(values, ElementsAre(42, 3, 8, 7));
    } else if (comm.rank() == 0) {
        auto recv_buf = comm.recv<int>();
        EXPECT_THAT(recv_buf, ElementsAre(42, 3, 8, 7));
    }
    comm.barrier();
}
TEST_F(ISendTest, send_buf_ownership_single_element) {
    Communicator comm;
    if (comm.size() < 2) {
        return;
    }
    if (comm.rank() == 1) {
        int  value = 42;
        auto req   = comm.isend(send_buf_out(std::move(value)), destination(0));
        value      = req.wait();
        EXPECT_EQ(value, 42);
    } else if (comm.rank() == 0) {
        auto recv_buf = comm.recv_single<int>();
        EXPECT_EQ(recv_buf, 42);
    }
    comm.barrier();
}
