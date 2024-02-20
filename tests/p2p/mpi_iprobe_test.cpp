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

#include <algorithm>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/has_member.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/iprobe.hpp"

using namespace ::kamping;

KAMPING_MAKE_HAS_MEMBER(extract_status)

class IProbeTest : public ::testing::Test {
    void SetUp() override {
        // this makes sure that messages don't spill from other tests
        MPI_Barrier(MPI_COMM_WORLD);
    }
    void TearDown() override {
        // this makes sure that messages don't spill to other tests
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

TEST_F(IProbeTest, direct_probe_with_status_out) {
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
            // return status
            auto result = comm.iprobe(source(other), tag(asserting_cast<int>(other)), status_out());
            while (!result.has_value()) {
                result = comm.iprobe(source(other), tag(asserting_cast<int>(other)), status_out());
            }
            EXPECT_TRUE(has_member_extract_status_v<decltype(result.value())>);
            auto status = result->extract_status();
            EXPECT_EQ(status.source(), other);
            EXPECT_EQ(status.tag(), other);
            EXPECT_EQ(status.count<int>(), other);
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

TEST_F(IProbeTest, direct_probe_with_wrapped_status) {
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
            // wrapped status
            Status kmp_status;
            while (!comm.iprobe(source(other), tag(asserting_cast<int>(other)), status_out(kmp_status))) {
            }
            EXPECT_EQ(kmp_status.source(), other);
            EXPECT_EQ(kmp_status.tag(), other);
            EXPECT_EQ(kmp_status.count<int>(), other);
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

TEST_F(IProbeTest, direct_probe_with_native_status) {
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
            // native status
            MPI_Status mpi_status;
            while (!comm.iprobe(source(other), tag(asserting_cast<int>(other)), status_out(mpi_status))) {
            }
            EXPECT_EQ(mpi_status.MPI_SOURCE, other);
            EXPECT_EQ(mpi_status.MPI_TAG, other);
            int count;
            MPI_Get_count(&mpi_status, MPI_INT, &count);
            EXPECT_EQ(count, other);
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

TEST_F(IProbeTest, direct_probe_with_implicit_ignore_status) {
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
            while (!comm.iprobe(source(other), tag(asserting_cast<int>(other)))) {
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

TEST_F(IProbeTest, direct_probe_with_explicit_ignore_status) {
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
            // ignore status
            while (!comm.iprobe(source(other), tag(asserting_cast<int>(other)), status(kamping::ignore<>))) {
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

TEST_F(IProbeTest, explicit_any_source_probe) {
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
            // explicit any source probe
            auto result = comm.iprobe(source(rank::any), tag(asserting_cast<int>(other)), status_out());
            while (!result.has_value()) {
                result = comm.iprobe(source(rank::any), tag(asserting_cast<int>(other)), status_out());
            }
            auto status = result->extract_status();
            EXPECT_EQ(status.source(), other);
            EXPECT_EQ(status.tag(), other);
            EXPECT_EQ(status.count<int>(), other);
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

TEST_F(IProbeTest, implicit_any_source_probe) {
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
            // implicit any source probe
            auto result = comm.iprobe(tag(asserting_cast<int>(other)), status_out());
            while (!result.has_value()) {
                result = comm.iprobe(tag(asserting_cast<int>(other)), status_out());
            }
            auto status = result->extract_status();
            EXPECT_EQ(status.source(), other);
            EXPECT_EQ(status.tag(), other);
            EXPECT_EQ(status.count<int>(), other);

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

TEST_F(IProbeTest, explicit_any_tag_probe) {
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
            // explicit any tag probe
            auto result = comm.iprobe(source(other), tag(tags::any), status_out());
            while (!result.has_value()) {
                result = comm.iprobe(source(other), tag(tags::any), status_out());
            }
            auto status = result->extract_status();
            EXPECT_EQ(status.source(), other);
            EXPECT_EQ(status.tag(), other);
            EXPECT_EQ(status.count<int>(), other);

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

TEST_F(IProbeTest, implicit_any_tag_probe) {
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
            // implicit any tag probe
            auto result = comm.iprobe(source(other), status_out());
            while (!result.has_value()) {
                result = comm.iprobe(source(other), status_out());
            }
            auto status = result->extract_status();
            EXPECT_EQ(status.source(), other);
            EXPECT_EQ(status.tag(), other);
            EXPECT_EQ(status.count<int>(), other);

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

TEST_F(IProbeTest, explicit_arbitrary_probe) {
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
            auto result = comm.iprobe(source(rank::any), tag(tags::any), status_out());
            while (!result.has_value()) {
                result = comm.iprobe(source(rank::any), tag(tags::any), status_out());
            }
            auto status = result->extract_status();
            auto source = status.source();
            EXPECT_FALSE(received_message_from[source]);
            EXPECT_EQ(status.tag(), status.source_signed());
            EXPECT_EQ(status.count_signed<int>(), source);

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
        EXPECT_TRUE(std::all_of(received_message_from.begin(), received_message_from.end(), [](bool const& received) {
            return received;
        }));
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IProbeTest, implicit_arbitrary_probe) {
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
            auto result = comm.iprobe(status_out());
            while (!result.has_value()) {
                result = comm.iprobe(status_out());
            }
            auto status = result->extract_status();
            auto source = status.source();
            EXPECT_FALSE(received_message_from[source]);
            EXPECT_EQ(status.tag(), status.source_signed());
            EXPECT_EQ(status.count_signed<int>(), source);

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
        EXPECT_TRUE(std::all_of(received_message_from.begin(), received_message_from.end(), [](bool const& received) {
            return received;
        }));
    }
    // ensure that we have received all inflight messages
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

TEST_F(IProbeTest, probe_null) {
    Communicator comm;
    auto         result = comm.iprobe(source(rank::null), status_out());
    while (!result.has_value()) {
        result = comm.iprobe(source(rank::null), status_out());
    }
    auto status = result->extract_status();
    EXPECT_EQ(status.source_signed(), MPI_PROC_NULL);
    EXPECT_EQ(status.tag(), MPI_ANY_TAG);
    EXPECT_EQ(status.count<int>(), 0);
}

TEST_F(IProbeTest, probe_null_structured_binding) {
    Communicator comm;
    auto         result = comm.iprobe(source(rank::null), status_out());
    while (!result.has_value()) {
        result = comm.iprobe(source(rank::null), status_out());
    }
    auto const& [status] = *result;
    EXPECT_EQ(status.source_signed(), MPI_PROC_NULL);
    EXPECT_EQ(status.tag(), MPI_ANY_TAG);
    EXPECT_EQ(status.count<int>(), 0);
}

TEST_F(IProbeTest, nothing_to_probe) {
    Communicator comm;
    EXPECT_FALSE(comm.iprobe());
}

TEST_F(IProbeTest, nothing_to_probe_with_status) {
    Communicator comm;
    EXPECT_FALSE(comm.iprobe(status_out()).has_value());
}
