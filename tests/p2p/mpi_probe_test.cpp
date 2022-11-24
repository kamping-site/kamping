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

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/probe.hpp"

using namespace ::kamping;

TEST(ProbeTest, direct_probe) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;
    MPI_Isend(v.data(), asserting_cast<int>(v.size()), MPI_INT, 0, comm.rank_signed(), comm.mpi_communicator(), &req);
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
        }
    }
}

TEST(ProbeTest, any_source_probe) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;
    MPI_Isend(v.data(), asserting_cast<int>(v.size()), MPI_INT, 0, comm.rank_signed(), comm.mpi_communicator(), &req);
    MPI_Barrier(comm.mpi_communicator());
    if (comm.rank() == 0) {
        for (auto other = comm.size_signed() - 1; other >= 0; other--) {
            {
                auto status = comm.probe(source(rank::any), tag(asserting_cast<int>(other)), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
            {
                auto status = comm.probe(tag(asserting_cast<int>(other)), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
        }
    }
}

TEST(ProbeTest, any_tag_probe) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;
    MPI_Isend(v.data(), asserting_cast<int>(v.size()), MPI_INT, 0, comm.rank_signed(), comm.mpi_communicator(), &req);
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            {
                auto status = comm.probe(source(other), tag(tags::any), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
            {
                auto status = comm.probe(source(other), status_out()).status();
                ASSERT_EQ(status.source(), other);
                ASSERT_EQ(status.tag(), other);
                ASSERT_EQ(status.count<int>(), other);
            }
        }
    }
}

TEST(ProbeTest, arbitrary_probe) {
    Communicator     comm;
    std::vector<int> v(comm.rank(), 42);
    MPI_Request      req;
    MPI_Isend(v.data(), asserting_cast<int>(v.size()), MPI_INT, 0, comm.rank_signed(), comm.mpi_communicator(), &req);
    if (comm.rank() == 0) {
        for (size_t other = 0; other < comm.size(); other++) {
            {
                auto status = comm.probe(source(rank::any), tag(tags::any), status_out()).status();
                auto source = status.source_signed();
                ASSERT_EQ(status.tag(), source);
                ASSERT_EQ(status.count_signed<int>(), source);
            }
            {
                auto status = comm.probe(status_out()).status();
                auto source = status.source_signed();
                ASSERT_EQ(status.tag(), source);
                ASSERT_EQ(status.count_signed<int>(), source);
            }
        }
    }
}

TEST(ProbeTest, probe_null) {
    Communicator comm;
    auto         status = comm.probe(source(rank::null), status_out()).status();
    ASSERT_EQ(status.source_signed(), MPI_PROC_NULL);
    ASSERT_EQ(status.tag(), MPI_ANY_TAG);
    ASSERT_EQ(status.count<int>(), 0);
}
