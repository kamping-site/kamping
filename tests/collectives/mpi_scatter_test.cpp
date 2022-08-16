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

#include "../test_assertions.hpp"

#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

namespace {
std::vector<int> create_input_vector_on_root(Communicator const& comm, int const elements_per_rank, int root = -1) {
    if (root < 0) {
        root = comm.root_signed();
    }

    std::vector<int> input;
    if (comm.rank_signed() == root) {
        input.resize(static_cast<std::size_t>(elements_per_rank) * comm.size());
        for (int rank = 0; rank < comm.size_signed(); ++rank) {
            auto begin = input.begin() + rank * elements_per_rank;
            auto end   = begin + elements_per_rank;
            std::fill(begin, end, rank);
        }
    }
    return input;
}
} // namespace

TEST(ScatterTest, scatter_single_element_no_recv_buffer) {
    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = comm.scatter(send_buf(input)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_single_element_with_recv_buffer) {
    Communicator comm;

    auto const       input = create_input_vector_on_root(comm, 1);
    std::vector<int> result;
    comm.scatter(send_buf(input), recv_buf(result));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_single_element_with_recv_count) {
    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = comm.scatter(send_buf(input), recv_counts(1)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_extract_recv_count) {
    Communicator comm;

    auto const input = create_input_vector_on_root(comm, 1);

    EXPECT_EQ(comm.scatter(send_buf(input)).extract_recv_counts(), 1);

    int recv_count_value;
    comm.scatter(send_buf(input), recv_counts_out(recv_count_value));
    EXPECT_EQ(recv_count_value, 1);
}

TEST(ScatterTest, scatter_multiple_elements) {
    int const elements_per_pe = 4;

    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, elements_per_pe);
    auto const result = comm.scatter(send_buf(input)).extract_recv_buffer();

    ASSERT_EQ(result.size(), elements_per_pe);
    EXPECT_THAT(result, Each(comm.rank()));
}

TEST(ScatterTest, scatter_with_send_buf_only_on_root_with_recv_buf) {
    Communicator comm;

    auto const       input = create_input_vector_on_root(comm, 1);
    std::vector<int> result;
    if (comm.is_root()) {
        comm.scatter(send_buf(input), recv_buf(result));
    } else {
        comm.scatter(send_buf(ignore<int>), recv_buf(result));
    }

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_send_buf_only_on_root) {
    Communicator comm;

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = (comm.is_root()) ? comm.scatter(send_buf(input)).extract_recv_buffer()
                                         : comm.scatter(send_buf(ignore<int>)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_root_arg) {
    Communicator comm;
    int const    root = comm.size_signed() - 1; // use last PE as root

    auto const input  = create_input_vector_on_root(comm, 1, root);
    auto const result = comm.scatter(send_buf(input), kamping::root(root)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_nonzero_root_comm) {
    Communicator comm;
    comm.root(comm.size() - 1);

    auto const input  = create_input_vector_on_root(comm, 1);
    auto const result = comm.scatter(send_buf(input)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_recv_count_out) {
    Communicator comm;

    auto const input = create_input_vector_on_root(comm, 2);
    int        recv_count;
    auto const result = comm.scatter(send_buf(input), recv_counts_out(recv_count)).extract_recv_buffer();

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(recv_count, 2);
}

TEST(ScatterTest, scatter_with_custom_sendbuf_and_type) {
    Communicator comm;
    struct Data {
        int value;
    };

    ::testing::OwnContainer<Data> input(static_cast<std::size_t>(comm.size()));
    if (comm.is_root()) {
        for (size_t rank = 0; rank < comm.size(); ++rank) {
            input[rank].value = asserting_cast<int>(rank);
        }
    }

    auto const result = comm.scatter(send_buf(input)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front().value, comm.rank());
}

TEST(ScatterTest, scatter_with_nonempty_sendbuf_on_non_root) {
    Communicator comm;

    std::vector<int> input(static_cast<std::size_t>(comm.size()));
    for (size_t rank = 0; rank < comm.size(); ++rank) {
        input[rank] = asserting_cast<int>(rank);
    }

    auto const result = comm.scatter(send_buf(input)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_different_roots_on_different_processes) {
    Communicator comm;
    auto const   input = create_input_vector_on_root(comm, 1);
    if (comm.size() > 1) {
        EXPECT_KASSERT_FAILS(comm.scatter(send_buf(input), root(comm.rank())), "");
    }
}
