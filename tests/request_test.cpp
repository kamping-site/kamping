// This file is part of KaMPIng.
//
// Copyright 2021-2023 The KaMPIng Authors
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

#include "kamping/communicator.hpp"
#include "kamping/request.hpp"

static bool                  test_succeed    = false;
static MPI_Request           handled_request = MPI_REQUEST_NULL;
static std::set<MPI_Request> handled_requests;

int MPI_Wait(MPI_Request* request, MPI_Status*) {
    // This doesn't actually call MPI_Wait on the request, so we can just pass
    // arbitrary requests object and see if they get passed correctly to the MPI
    // call.
    handled_request = *request;
    return MPI_SUCCESS;
}

int MPI_Test(MPI_Request* request, int* flag, MPI_Status*) {
    // This doesn't actually call MPI_Test on the request, so we can mock a failing test.
    handled_request = *request;
    *flag           = test_succeed;
    return MPI_SUCCESS;
}

int MPI_Waitall(int count, MPI_Request* array_of_requests, MPI_Status* array_of_statuses) {
    auto begin = array_of_requests;
    auto end   = array_of_requests + count;
    for (auto current = begin; current != end; current++) {
        handled_requests.insert(*current);
    }
    return PMPI_Waitall(count, array_of_requests, array_of_statuses);
}

class RequestTest : public ::testing::Test {
    void SetUp() override {
        test_succeed    = false;
        handled_request = MPI_REQUEST_NULL;
        handled_requests.clear();
    }
    void TearDown() override {
        test_succeed    = false;
        handled_request = MPI_REQUEST_NULL;
        handled_requests.clear();
    }
};

TEST_F(RequestTest, construction) {
    {
        kamping::Request req;
        EXPECT_EQ(req.mpi_request(), MPI_REQUEST_NULL);
    }
    {
        // trivial uninitialized request
        MPI_Request      request{};
        kamping::Request req(request);
        EXPECT_EQ(req.mpi_request(), request);
    }
    {
        // actual request
        MPI_Request mpi_req;
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &mpi_req);
        EXPECT_NE(mpi_req, MPI_REQUEST_NULL);
        kamping::Request req{mpi_req};
        EXPECT_EQ(req.mpi_request(), mpi_req);
        // we are intercepting the calls to MPI_Wait and do nothing useful there, so we have to call the PMPI variant
        // explicitely here
        PMPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
    }
}

TEST_F(RequestTest, wait) {
    {
        // trivial uninitialized request
        MPI_Request      request{};
        kamping::Request req(request);
        req.wait();
        EXPECT_EQ(handled_request, request);
    }
    {
        // actual request
        MPI_Request mpi_req;
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &mpi_req);
        EXPECT_NE(mpi_req, MPI_REQUEST_NULL);
        kamping::Request req{mpi_req};
        req.wait();
        EXPECT_EQ(handled_request, mpi_req);
        // we are intercepting the calls to MPI_Wait and do nothing useful there, so we have to call the PMPI variant
        // explicitely here
        PMPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
    }
}

TEST_F(RequestTest, test_succeed) {
    {
        test_succeed = true;
        MPI_Request      request{};
        kamping::Request req(request);
        EXPECT_TRUE(req.test());
        EXPECT_EQ(handled_request, request);
    }
    {
        // actual request
        MPI_Request mpi_req;
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &mpi_req);
        EXPECT_NE(mpi_req, MPI_REQUEST_NULL);
        kamping::Request req{mpi_req};
        test_succeed = true;
        EXPECT_TRUE(req.test());
        // we are intercepting the calls to MPI_Wait and do nothing useful there, so we have to call the PMPI variant
        // explicitely here
        PMPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
    }
}

TEST_F(RequestTest, test_fail) {
    {
        test_succeed = false;
        MPI_Request      request{};
        kamping::Request req(request);
        EXPECT_FALSE(req.test());
        EXPECT_EQ(handled_request, request);
    }
    {
        // actual request
        MPI_Request mpi_req;
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &mpi_req);
        EXPECT_NE(mpi_req, MPI_REQUEST_NULL);
        kamping::Request req{mpi_req};
        test_succeed = false;
        EXPECT_FALSE(req.test());
        // we are intercepting the calls to MPI_Wait and do nothing useful there, so we have to call the PMPI variant
        // explicitely here
        PMPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
    }
}

TEST_F(RequestTest, unsafe_wait_all_container) {
    std::vector<kamping::Request> requests(3);
    for (auto& request: requests) {
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &request.mpi_request());
    }
    std::set<MPI_Request> expected_requests{
        requests[0].mpi_request(),
        requests[1].mpi_request(),
        requests[2].mpi_request()};
    kamping::requests::wait_all_with_undefined_behavior(requests);
    EXPECT_EQ(handled_requests, expected_requests);
}

TEST_F(RequestTest, unsafe_wait_all_container_moved) {
    std::vector<kamping::Request> requests(3);
    for (auto& request: requests) {
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &request.mpi_request());
    }
    std::set<MPI_Request> expected_requests{
        requests[0].mpi_request(),
        requests[1].mpi_request(),
        requests[2].mpi_request()};
    kamping::requests::wait_all_with_undefined_behavior(std::move(requests));
    EXPECT_EQ(handled_requests, expected_requests);
}

TEST_F(RequestTest, wait_all_container) {
    std::vector<kamping::Request> requests(3);
    for (auto& request: requests) {
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &request.mpi_request());
    }
    std::set<MPI_Request> expected_requests{
        requests[0].mpi_request(),
        requests[1].mpi_request(),
        requests[2].mpi_request()};
    kamping::requests::wait_all(requests);
    EXPECT_EQ(handled_requests, expected_requests);
}

TEST_F(RequestTest, wait_all_container_moved) {
    std::vector<kamping::Request> requests(3);
    for (auto& request: requests) {
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &request.mpi_request());
    }
    std::set<MPI_Request> expected_requests{
        requests[0].mpi_request(),
        requests[1].mpi_request(),
        requests[2].mpi_request()};
    kamping::requests::wait_all(std::move(requests));
    EXPECT_EQ(handled_requests, expected_requests);
}

TEST_F(RequestTest, wait_all_container_native) {
    std::vector<MPI_Request> requests(3);
    for (auto& request: requests) {
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &request);
    }
    std::set<MPI_Request> expected_requests{requests.begin(), requests.end()};
    kamping::requests::wait_all(requests);
    EXPECT_EQ(handled_requests, expected_requests);
}

TEST_F(RequestTest, wait_all_container_native_moved) {
    std::vector<MPI_Request> requests(3);
    for (auto& request: requests) {
        MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &request);
    }
    std::set<MPI_Request> expected_requests{requests.begin(), requests.end()};
    kamping::requests::wait_all(std::move(requests));
    EXPECT_EQ(handled_requests, expected_requests);
}

TEST_F(RequestTest, wait_all_variadic) {
    kamping::Request req1;
    kamping::Request req2;
    MPI_Request      req3;

    MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &req1.mpi_request());
    MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &req2.mpi_request());
    MPI_Ibarrier(kamping::comm_world().mpi_communicator(), &req3);

    std::set<MPI_Request> expected_requests{req1.mpi_request(), req2.mpi_request(), req3};
    kamping::requests::wait_all(req1, std::move(req2), req3);
    EXPECT_EQ(handled_requests, expected_requests);
}
