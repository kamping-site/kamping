
// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <numeric>

#include "../helpers_for_testing.hpp"
#include "kamping/communicator.hpp"


using namespace ::kamping;
using namespace ::testing;

TEST(BcastTest, SingleElement) {
    Communicator comm;

    // Basic use case, broadcast a single POD.
    int value = comm.rank();
    comm.bcast(send_recv_buf(value));
    EXPECT_EQ(value, comm.root());

    // The following test is only valid if the communicator has more than one rank.
    if (comm.size() >= 2) {
        // Broadcast a single POD to all processes, manually specify the root process.
        value = comm.rank();
        comm.bcast(send_recv_buf(value), root(1));
        EXPECT_EQ(value, 1);

        // Broadcast a single POD to all processes, use a non-default communicator's root.
        value              = comm.rank();
        const int new_root = 1;
        comm.root(new_root);
        ASSERT_EQ(new_root, comm.root());
        comm.bcast(send_recv_buf(value));
        EXPECT_EQ(value, new_root);
    }
}

TEST(BcastTest, Vector) {
    Communicator comm;

    std::vector<int> values(4);
    if (comm.is_root()) {
        std::fill(values.begin(), values.end(), comm.rank());
    }

    comm.bcast(send_recv_buf(values));
    EXPECT_THAT(values, Each(Eq(comm.root())));
}
