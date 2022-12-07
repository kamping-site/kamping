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

#include <algorithm>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/probe.hpp"

using namespace ::kamping;

TEST(ProbeTest, direct_probe) {
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
            {
                // return status
                auto status = comm.probe(source(other), tag(asserting_cast<int>(other)), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
            {
                // wrapped status
                Status kmp_status;
                comm.probe(source(other), tag(asserting_cast<int>(other)), status(kmp_status));
                ASSERT_EQ(kmp_status.source(), other);
                ASSERT_EQ(kmp_status.tag(), other);
                ASSERT_EQ(kmp_status.count<int>(), other);
            }
            {
                // native status
                MPI_Status mpi_status;
                comm.probe(source(other), tag(asserting_cast<int>(other)), status(mpi_status));
                ASSERT_EQ(mpi_status.MPI_SOURCE, other);
                ASSERT_EQ(mpi_status.MPI_TAG, other);
                int count;
                MPI_Get_count(&mpi_status, MPI_INT, &count);
                ASSERT_EQ(count, other);
            }
            {
                // ignore status
                comm.probe(source(other), tag(asserting_cast<int>(other)));
                ASSERT_TRUE(true);
                comm.probe(source(other), tag(asserting_cast<int>(other)), status(kamping::ignore<>));
                ASSERT_TRUE(true);
            }
            std::vector<int> recv_buf(other);
            MPI_Recv(
                recv_buf.data(),            // recv_buf
                asserting_cast<int>(other), // recv_size
                MPI_INT,                    // recv_type
                asserting_cast<int>(other), // source
                asserting_cast<int>(other), // tag
                MPI_COMM_WORLD,             // comm
                MPI_STATUS_IGNORE           // status
            );
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST(ProbeTest, any_source_probe) {
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
    MPI_Barrier(comm.mpi_communicator());
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            {
                // explicit any source probe
                auto status = comm.probe(source(rank::any), tag(asserting_cast<int>(other)), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
            {
                // implicit any source probe
                auto status = comm.probe(tag(asserting_cast<int>(other)), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
            std::vector<int> recv_buf(other);
            MPI_Recv(
                recv_buf.data(),            // recv_buf
                asserting_cast<int>(other), // recv_size
                MPI_INT,                    // recv_type
                asserting_cast<int>(other), // source
                asserting_cast<int>(other), // tag
                MPI_COMM_WORLD,             // comm
                MPI_STATUS_IGNORE           // status
            );
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST(ProbeTest, any_tag_probe) {
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
            {
                // explicit any tag probe
                auto status = comm.probe(source(other), tag(tags::any), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
            {
                // implicit any tag probe
                auto status = comm.probe(source(other), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
            std::vector<int> recv_buf(other);
            MPI_Recv(
                recv_buf.data(),            // recv_buf
                asserting_cast<int>(other), // recv_size
                MPI_INT,                    // recv_type
                asserting_cast<int>(other), // source
                asserting_cast<int>(other), // tag
                MPI_COMM_WORLD,             // comm
                MPI_STATUS_IGNORE           // status
            );
        }
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST(ProbeTest, arbitrary_probe) {
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
        // because we may receive arbitrary message, we keep track of them
        std::vector<bool> received_message_from(comm.size());

        for (size_t other = 0; other < comm.size(); other++) {
            auto status = comm.probe(source(rank::any), tag(tags::any), status_out()).status();
            auto source = status.source();
            ASSERT_FALSE(received_message_from[source]);
            ASSERT_EQ(status.tag(), status.source_signed());
            ASSERT_EQ(status.count_signed<int>(), source);

            std::vector<int> recv_buf(source);
            MPI_Recv(
                recv_buf.data(),             // recv_buf
                asserting_cast<int>(source), // recv_size
                MPI_INT,                     // recv_type
                asserting_cast<int>(source), // source
                asserting_cast<int>(source), // tag
                MPI_COMM_WORLD,              // comm
                MPI_STATUS_IGNORE            // status
            );
            received_message_from[source] = true;
        }
        // check that we probed all messages
        ASSERT_TRUE(std::all_of(received_message_from.begin(), received_message_from.end(), [](bool const& received) {
            return received;
        }));
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    // again with implicit any probe
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
        // because we may receive arbitrary message, we keep track of them
        std::vector<bool> received_message_from(comm.size());

        for (size_t other = 0; other < comm.size(); other++) {
            auto status = comm.probe(status_out()).status();
            auto source = status.source();
            ASSERT_FALSE(received_message_from[source]);
            ASSERT_EQ(status.tag(), status.source_signed());
            ASSERT_EQ(status.count_signed<int>(), source);

            std::vector<int> recv_buf(source);
            MPI_Recv(
                recv_buf.data(),             // recv_buf
                asserting_cast<int>(source), // recv_size
                MPI_INT,                     // recv_type
                asserting_cast<int>(source), // source
                asserting_cast<int>(source), // tag
                MPI_COMM_WORLD,              // comm
                MPI_STATUS_IGNORE            // status
            );
            received_message_from[source] = true;
        }
        // check that we probed all messages
        ASSERT_TRUE(std::all_of(received_message_from.begin(), received_message_from.end(), [](bool const& received) {
            return received;
        }));
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST(ProbeTest, probe_null) {
    Communicator comm;
    auto         status = comm.probe(source(rank::null), status_out()).status();
    ASSERT_EQ(status.source_signed(), MPI_PROC_NULL);
    ASSERT_EQ(status.tag(), MPI_ANY_TAG);
    ASSERT_EQ(status.count<int>(), 0);
}
