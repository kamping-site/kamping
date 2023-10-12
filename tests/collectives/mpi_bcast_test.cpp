// This file is part of KaMPIng.
//
// Copyright 2022-2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "../test_assertions.hpp"

#include <numeric>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/assertion_levels.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters.hpp"
#include "kassert/kassert.hpp"

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
    size_t const root = comm.size() - 1;
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
    value = comm.rank();
    comm.bcast(send_recv_buf(value), send_recv_count(1));
    EXPECT_EQ(value, root);
}

TEST(BcastTest, extract_receive_buffer) {
    Communicator comm;

    // Basic use case, broadcast a single POD.
    std::vector<size_t> values;
    if (comm.is_root()) {
        values = {42, 1337};
        comm.bcast(send_recv_buf(values));
    } else {
        values = comm.bcast(send_recv_buf(alloc_new<std::vector<size_t>>)).extract_recv_buffer();
    }

    EXPECT_THAT(values, ElementsAre(42, 1337));
}

TEST(BcastTest, single_element_bool) {
    Communicator comm;

    // Basic use case, broadcast a single POD.

    bool value;
    if (comm.is_root()) {
        value = true;
    } else {
        value = false;
    }
    comm.bcast(send_recv_buf(value));
    EXPECT_EQ(value, true);
}

TEST(Bcasttest, vector_partial_transfer) {
    Communicator comm;

    std::vector<int> values(5);
    int              num_transferred_values = 3;
    std::iota(values.begin(), values.end(), comm.rank() * 10);
    kamping::Span<int> transfer_view(values.data(), asserting_cast<size_t>(num_transferred_values));

    comm.bcast(send_recv_buf(transfer_view));
    EXPECT_EQ(values.size(), 5);
    EXPECT_THAT(values, ElementsAre(0, 1, 2, comm.rank() * 10 + 3, comm.rank() * 10 + 4));

    std::iota(values.begin(), values.end(), comm.rank() * 10);
    comm.bcast(send_recv_buf(transfer_view), send_recv_count(num_transferred_values));
    EXPECT_EQ(values.size(), 5);
    EXPECT_THAT(values, ElementsAre(0, 1, 2, comm.rank() * 10 + 3, comm.rank() * 10 + 4));

    std::iota(values.begin(), values.end(), comm.rank() * 10);
    num_transferred_values = -1;
    comm.bcast(send_recv_buf(transfer_view), send_recv_count_out(num_transferred_values));
    EXPECT_EQ(values.size(), 5);
    EXPECT_EQ(num_transferred_values, 3);
    EXPECT_THAT(values, ElementsAre(0, 1, 2, comm.rank() * 10 + 3, comm.rank() * 10 + 4));
}

TEST(BcastTest, vector_send_recv_count_deduction) {
    Communicator comm;

    { // send_recv_count is inferred from the size of the buffer at root.
        size_t const num_values = 4;

        std::vector<int> values;
        if (comm.is_root()) {
            values.resize(num_values);
            std::fill(values.begin(), values.end(), comm.rank());
        }

        int count = -1;
        comm.bcast(send_recv_buf<resize_to_fit>(values), send_recv_count_out(count));
        EXPECT_EQ(count, num_values);
        EXPECT_EQ(values.size(), num_values);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }
    { // All ranks provide the same send_recv_count.
        size_t const num_values = 4;

        std::vector<int> values(num_values);
        if (comm.is_root()) {
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf(values), send_recv_count(asserting_cast<int>(num_values)));
        EXPECT_EQ(values.size(), num_values);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    if (comm.size() > 1) {
        { // Some ranks provide a send_recv_count, some don't. This is not allowed
            size_t const num_values = 4;

            std::vector<int> values(num_values);
            if (comm.is_root()) {
                EXPECT_KASSERT_FAILS(
                    comm.bcast(send_recv_buf(values), send_recv_count(asserting_cast<int>(num_values))),
                    ""
                );
            } else {
                EXPECT_KASSERT_FAILS(comm.bcast(send_recv_buf(values)), "");
            }
        }
        { // Root rank provides send_recv_count, the other ranks request as an out parameter. This should fail, explicit
          // counts must be present either on all or no ranks.
            comm.root(0);
            std::vector<int> values(0);
            int              num_elements = 43;

            if (comm.is_root()) {
                values.resize(asserting_cast<size_t>(num_elements));
                EXPECT_KASSERT_FAILS(comm.bcast(send_recv_buf(values), send_recv_count(num_elements)), "");
            } else {
                values.resize(comm.rank());
                [[maybe_unused]] int num_elements_received = -1;
                EXPECT_KASSERT_FAILS(comm.bcast(send_recv_buf(values), send_recv_count_out(num_elements_received)), "");
            }
        }
    }
#endif
}

TEST(BcastTest, vector_default_resize_policy_should_be_no_resize) {
    Communicator comm;

    { // all large buffers are large enough and are not resized
        std::vector<int> values(4 + comm.rank() + 2, -1);
        if (comm.is_root()) {
            std::fill(values.begin(), values.begin() + 4, comm.rank());
        }

        comm.bcast(send_recv_buf(values), send_recv_count(4));
        EXPECT_EQ(values.size(), 4 + comm.rank() + 2);
        EXPECT_THAT(Span<int>(values.data(), 4), Each(Eq(comm.root())));
        EXPECT_THAT(Span<int>(values.data() + 4, values.size() - 4), Each(Eq(-1)));
    }
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
    { // buffer on receiving side too small
        std::vector<int> values;
        if (comm.is_root()) {
            values.resize(100);
            std::fill(values.begin(), values.end(), comm.rank());
        } else {
            values.resize(0);
        }

        if (!comm.is_root()) {
            EXPECT_KASSERT_FAILS(comm.bcast(send_recv_buf(values)), "");
        } else {
            comm.bcast(send_recv_buf(values));
        }
    }
#endif
}

TEST(BcastTest, vector_resize_policy_no_resize) {
    Communicator comm;

    { // all large buffers are large enough and are not resized
        std::vector<int> values(4 + comm.rank() + 2, -1);
        if (comm.is_root()) {
            std::fill(values.begin(), values.begin() + 4, comm.rank());
        }

        comm.bcast(send_recv_buf<no_resize>(values), send_recv_count(4));
        EXPECT_EQ(values.size(), 4 + comm.rank() + 2);
        EXPECT_THAT(Span<int>(values.data(), 4), Each(Eq(comm.root())));
        EXPECT_THAT(Span<int>(values.data() + 4, values.size() - 4), Each(Eq(-1)));
    }
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
    { // buffer on receiving side too small
        std::vector<int> values;
        if (comm.is_root()) {
            values.resize(100);
            std::fill(values.begin(), values.end(), comm.rank());
        } else {
            values.resize(0);
        }

        if (!comm.is_root()) {
            EXPECT_KASSERT_FAILS(comm.bcast(send_recv_buf<no_resize>(values)), "");
        } else {
            comm.bcast(send_recv_buf<no_resize>(values));
        }
    }
#endif
}

TEST(BcastTest, vector_resize_policy_grow) {
    Communicator comm;

    { // buffers which are large enough are not resized
        std::vector<int> values(4 + comm.rank(), -1);
        if (comm.is_root()) {
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf<grow_only>(values), send_recv_count(4));
        EXPECT_EQ(values.size(), 4 + comm.rank());
        EXPECT_THAT(Span<int>(values.data(), 4), Each(Eq(comm.root())));
        EXPECT_THAT(Span<int>(values.data() + 4, values.size() - 4), Each(Eq(-1)));
    }
    { // buffers which are too small are resized
        std::vector<int> values(1);
        if (comm.is_root()) {
            values.resize(4);
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf<grow_only>(values), send_recv_count(4));
        EXPECT_EQ(values.size(), 4);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }
}

TEST(BcastTest, vector_resize_to_fit) {
    Communicator comm;

    { // buffers which are large enough are not resized
        std::vector<int> values(4 + comm.rank(), -1);
        if (comm.is_root()) {
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf<grow_only>(values), send_recv_count(4));
        EXPECT_EQ(values.size(), 4 + comm.rank());
        EXPECT_THAT(Span<int>(values.data(), 4), Each(Eq(comm.root())));
        EXPECT_THAT(Span<int>(values.data() + 4, values.size() - 4), Each(Eq(-1)));
    }
    { // buffers which are too small are resized
        std::vector<int> values(1);
        if (comm.is_root()) {
            values.resize(4);
            std::fill(values.begin(), values.end(), comm.rank());
        }

        comm.bcast(send_recv_buf<grow_only>(values), send_recv_count(4));
        EXPECT_EQ(values.size(), 4);
        EXPECT_THAT(values, Each(Eq(comm.root())));
    }
}

TEST(BcastTest, message_of_size_0) {
    Communicator comm;

    std::vector<int> values(0);
    EXPECT_NO_THROW(comm.bcast(send_recv_buf(values)));
    EXPECT_EQ(values.size(), 0);
}

TEST(BcastTest, send_recv_buf_parameter_only_on_root) {
    Communicator<OwnContainer> comm;

    OwnContainer<int> message;
    if (comm.is_root()) {
        message = {42, 1337};
        comm.bcast(send_recv_buf(message));
    } else {
        message = comm.bcast<int>().extract_recv_buffer();
    }
    EXPECT_THAT(message, ElementsAre(42, 1337));
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT_COMMUNICATION)
TEST(BcastTest, roots_differ) {
    Communicator comm;
    if (comm.size() > 1) {
        int val = comm.rank_signed();
        EXPECT_KASSERT_FAILS(
            comm.bcast(send_recv_buf(val), root(comm.rank())),
            "root() parameter must be the same on all ranks."
        );
    }
}
#endif

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
TEST(BcastTest, send_recv_buf_parameter_required_on_root) {
    Communicator comm;

    OwnContainer<int> message;
    EXPECT_KASSERT_FAILS(comm.bcast<int>(), "send_recv_buf must be provided on the root rank.");
}
#endif

TEST(BcastTest, bcast_single) {
    // bcast_single is a wrapper around bcast, providing the send_recv_count(1).
    // There is not much we can test here, that's not already tested by the tests for bcast.

    Communicator comm;

    int value = comm.rank_signed();
    EXPECT_NO_THROW(comm.bcast_single(send_recv_buf(value), root(0)));
    EXPECT_EQ(value, 0);

    std::vector<int> value_vector = {comm.rank_signed()};
    EXPECT_NO_THROW(comm.bcast_single(send_recv_buf(value_vector)));
    EXPECT_EQ(value_vector[0], 0);

    value_vector.resize(2);
    EXPECT_KASSERT_FAILS(comm.bcast_single(send_recv_buf(value_vector)), "");

    value_vector.resize(0);
    EXPECT_KASSERT_FAILS(comm.bcast_single(send_recv_buf(value_vector)), "");
}

TEST(BcastTest, bcast_single_send_recv_buf_parameter_only_on_root) {
    Communicator comm;

    int value = 1;
    if (comm.is_root()) {
        value = comm.rank_signed();
        comm.bcast_single(send_recv_buf(value));
    } else {
        value = comm.bcast_single<int>();
    }

    EXPECT_EQ(value, 0);
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT_COMMUNICATION)
TEST(BcastTest, bcast_single_send_recv_buf_parameter_required_on_root) {
    Communicator comm;

    OwnContainer<int> message;
    EXPECT_KASSERT_FAILS(comm.bcast_single<int>(), "send_recv_buf must be provided on the root rank.");
}
#endif

TEST(BcastTest, bcast_single_invalid_parameters) {
    Communicator comm;

    std::vector<int> input = {42, 1};

    EXPECT_KASSERT_FAILS(
        (comm.bcast_single(send_recv_buf(input))),
        "The send/receive buffer has to be of size 1 on all ranks."
    );
}
