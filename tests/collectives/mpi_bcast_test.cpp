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

    // Broadcast a single POD to all processes, manually specify the recv_count.
    /// @todo Uncomment, once we can check for failed assertions.
    value = comm.rank();
    // EXPECT_KASSERT_FAILS(comm.bcast(send_recv_buf(value), recv_count(0)));
    comm.bcast(send_recv_buf(value), recv_count(1));
    EXPECT_EQ(value, root);
    // EXPECT_KASSERT_FAILS(comm.bcast(send_recv_buf(value), recv_count(2)));
}

TEST(Bcasttest, vector_partial_transfer) {
    Communicator comm;

    std::vector<int> values(5);
    int              num_transferred_values = 3;
    std::iota(values.begin(), values.end(), comm.rank() * 10);
    kamping::Span<int> transfer_view(values.data(), asserting_cast<size_t>(num_transferred_values));

    /// @todo Once we can check assertions, check that providing an recv_count != transfer_view.size() fails.
    comm.bcast(send_recv_buf(transfer_view));
    EXPECT_EQ(values.size(), 5);
    EXPECT_THAT(values, ElementsAre(0, 1, 2, comm.rank() * 10 + 3, comm.rank() * 10 + 4));

    std::iota(values.begin(), values.end(), comm.rank() * 10);
    comm.bcast(send_recv_buf(transfer_view), recv_count(num_transferred_values));
    EXPECT_EQ(values.size(), 5);
    EXPECT_THAT(values, ElementsAre(0, 1, 2, comm.rank() * 10 + 3, comm.rank() * 10 + 4));

    std::iota(values.begin(), values.end(), comm.rank() * 10);
    num_transferred_values = -1;
    comm.bcast(send_recv_buf(transfer_view), recv_count_out(num_transferred_values));
    EXPECT_EQ(values.size(), 5);
    EXPECT_EQ(num_transferred_values, 3);
    EXPECT_THAT(values, ElementsAre(0, 1, 2, comm.rank() * 10 + 3, comm.rank() * 10 + 4));
}

TEST(BcastTest, vector_recv_count) {
    Communicator comm;

    { // All ranks provide the same recv_count.
        const size_t num_values = 4;

        std::vector<int> values(num_values);
        if (comm.is_root()) {
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf(values), recv_count(num_values));
        EXPECT_EQ(values.size(), num_values);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }

    {
        // Some buffer provide a recv_count, some don't.
        /// @todo Add test, once we can test failing assertions.
    }

    { // All buffers provide a recv_count, but they differ.
      /// @todo Add test, once we can test failing assertions.
    }
}

TEST(BcastTest, vector_recv_count_not_equal_to_vector_size) {
    Communicator comm;

    { // recv count < vector size
        const size_t num_values             = 4;
        const int    num_transferred_values = num_values - 1;

        std::vector<int> values(num_values);
        if (comm.is_root()) {
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf(values), recv_count(num_transferred_values));
        EXPECT_EQ(values.size(), num_transferred_values);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }

    // { // recv count > vector size
    //     const size_t num_values = 4;
    //     const int num_transferred_values = num_values + 1;

    //     std::vector<int> values(num_values);
    //     if (comm.is_root()) {
    //         std::fill(values.begin(), values.end(), comm.rank());
    //     }

    //     EXPECT_KASSERT_FAILS(comm.bcast(send_recv_buf(values), recv_count(num_transferred_values)));
    // }
}

TEST(BcastTest, vector_no_recv_count) {
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

TEST(BcastTest, vector_recv_count_as_out_parameter) {
    Communicator comm;

    { // All send_recv_bufs are already large enough.
        std::vector<int> values(4);
        if (comm.is_root()) {
            std::fill(values.begin(), values.end(), comm.rank());
        }

        int num_elements_received = -1;
        comm.bcast(send_recv_buf(values), recv_count_out(num_elements_received));
        EXPECT_EQ(values.size(), 4);
        EXPECT_EQ(num_elements_received, values.size());
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

        int num_elements_received = -1;
        comm.bcast(send_recv_buf(values), recv_count_out(num_elements_received));
        EXPECT_EQ(values.size(), 100);
        EXPECT_EQ(num_elements_received, values.size());
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

        int num_elements_received = -1;
        comm.bcast(send_recv_buf(values), recv_count_out(num_elements_received));
        EXPECT_EQ(values.size(), 43);
        EXPECT_EQ(num_elements_received, values.size());
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }
}

TEST(BcastTest, vector_needs_resizing_and_counts_are_given) {
    Communicator comm;

    size_t num_values = 10;

    std::vector<int> values;
    if (comm.is_root()) {
        values.resize(num_values);
        std::fill(values.begin(), values.end(), comm.rank());
    }
    comm.bcast(send_recv_buf(values), recv_count(asserting_cast<int>(num_values)));
    EXPECT_EQ(values.size(), num_values);
    EXPECT_THAT(values, Each(Eq(comm.root())));
}

TEST(BcastTest, message_of_size_0) {
    Communicator comm;

    std::vector<int> values(0);
    EXPECT_NO_THROW(comm.bcast(send_recv_buf(values)));
    EXPECT_EQ(values.size(), 0);

    values.resize(1);
    EXPECT_NO_THROW(comm.bcast(send_recv_buf(values), recv_count(0)));
    EXPECT_EQ(values.size(), 0);
}
