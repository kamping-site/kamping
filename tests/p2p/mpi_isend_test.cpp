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

#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/parameter_objects.hpp"

using namespace ::kamping;

// Note: These invariants tested here only hold when the tests are executed using more than one MPI rank!

static size_t call_hierarchy_level = 0;
static size_t send_counter         = 0;
static size_t bsend_counter        = 0;
static size_t ssend_counter        = 0;
static size_t rsend_counter        = 0;

class IsendTest : public ::testing::Test {
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

TEST_F(IsendTest, send_vector) {
    Communicator            comm;
    auto                    other_rank = (comm.root() + 1) % comm.size();
    static constexpr size_t n          = 100000000;
    if (comm.is_root()) {
        std::vector<int> values;
        values.resize(n);
        std::fill_n(values.begin(), n, 42);
        auto req = comm.isend(send_buf(values), destination(other_rank)).wait();
        int  x   = req;
        // ASSERT_EQ(send_counter, 1);
        // ASSERT_EQ(bsend_counter, 0);
        // ASSERT_EQ(ssend_counter, 0);
        // ASSERT_EQ(rsend_counter, 0);
    } else if (comm.rank() == other_rank) {
        std::vector<int> msg(n);
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
        // ASSERT_EQ(msg, (std::vector<int>{42, 3, 8, 7}));
        ASSERT_EQ(msg.size(), n);
        ASSERT_EQ(msg.back(), 42);
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    }
}
