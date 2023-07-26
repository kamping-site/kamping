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

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/collectives/ibarrier.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

std::set<MPI_Request> initialized_requests;
std::set<MPI_Request> completed_requests;
size_t                ibarrier_calls = 0;

int MPI_Wait(MPI_Request* request, MPI_Status* status) {
    if (*request != MPI_REQUEST_NULL) {
        completed_requests.insert(*request);
    }
    return PMPI_Wait(request, status);
}
int MPI_Ibarrier(MPI_Comm comm, MPI_Request* request) {
    auto errcode = PMPI_Ibarrier(comm, request);
    initialized_requests.insert(*request);
    ibarrier_calls++;
    return errcode;
}

class IBarrierTest : public ::testing::Test {
    void SetUp() override {
        ibarrier_calls = 0;
        initialized_requests.clear();
        completed_requests.clear();
    }
    void TearDown() override {
        ibarrier_calls = 0;
        EXPECT_EQ(initialized_requests, completed_requests);
        initialized_requests.clear();
        completed_requests.clear();
    }
};

TEST_F(IBarrierTest, ibarrier) {
    Communicator comm;
    auto         req = comm.ibarrier();
    req.wait();
    EXPECT_EQ(ibarrier_calls, 1);
}

TEST_F(IBarrierTest, ibarrier_non_owning_reference) {
    Communicator comm;
    Request      req;
    comm.ibarrier(request(req));
    req.wait();
    EXPECT_TRUE(true);
    EXPECT_EQ(ibarrier_calls, 1);
}

TEST_F(IBarrierTest, two_ibarriers) {
    Communicator comm;
    auto         req1 = comm.ibarrier();
    auto         req2 = comm.ibarrier();
    req1.wait();
    req2.wait();
    EXPECT_EQ(ibarrier_calls, 2);
}
