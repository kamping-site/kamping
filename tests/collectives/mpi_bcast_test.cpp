
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

#include <numeric>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/parameter_factories.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(BcastTest, single_element) {
    Communicator comm;

    // Basic use case, broadcast a single POD.
    size_t value = comm.rank();
    comm.bcast(send_recv_buf(value));
    EXPECT_EQ(value, comm.root());

    // TODO Using the unnamed first parameter.
    // value++;
    // comm.bcast(value);
    // EXPECT_EQ(value, comm.root() + 1);

    // Broadcast a single POD to all processes, manually specify the root process.
    assert(comm.size() > 0);
    const size_t root = comm.size() - 1;
    value             = comm.rank();
    comm.bcast(send_recv_buf(value), kamping::root(root));
    EXPECT_EQ(value, root);

    // Broadcast a single POD to all processes, use a non-default communicator's root.
    value = comm.rank();
    comm.root(root);
    ASSERT_EQ(root, comm.root());
    comm.bcast(send_recv_buf(value));
    EXPECT_EQ(value, root);
}

TEST(BcastTest, vector_send_recv_count) {
    Communicator comm;

    { // All ranks provide the same send_recv_count.
        const size_t num_values = 4;

        std::vector<int> values(num_values);
        if (comm.is_root()) {
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf(values), send_recv_count(num_values));
        EXPECT_EQ(values.size(), num_values);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }

    {
        // Some buffer provide a send_recv_count, some don't.
        /// @todo Add test, once we can test failing assertions.
    }

    { // All buffers provide a send_recv_count, but they differ.
      /// @todo Add test, once we can test failing assertions.
    }
}

TEST(BcastTest, vector_no_send_recv_count) {
    Communicator comm;

    { // All send_recv_bufs are already large enough.
        std::vector<int> values(4);
        if (comm.is_root()) {
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf(values));
        EXPECT_EQ(values.size(), 4);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }

    { // Some send_recv_bufs need to be resized.
        std::vector<int> values;
        if (comm.is_root()) {
            values.resize(100);
            std::fill(values.begin(), values.end(), comm.rank());
        } else {
            values.resize(0);
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf(values));
        EXPECT_EQ(values.size(), 100);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }

    { // All send_recv_bufs are of different size
        comm.root(0);
        std::vector<int> values;

        if (comm.is_root()) {
            values.resize(43);
            std::fill(values.begin(), values.end(), comm.rank());
        } else {
            values.resize(comm.rank());
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf(values));
        EXPECT_EQ(values.size(), 43);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }
}
