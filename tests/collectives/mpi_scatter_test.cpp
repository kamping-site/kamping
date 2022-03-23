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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

namespace {
std::vector<int> create_input_vector_on_root(Communicator const& comm, int const elements_per_pe) {
    std::vector<int> input;
    if (comm.is_root()) {
        input.resize(static_cast<std::size_t>(elements_per_pe * comm.size()));
        for (int pe = 0; pe < comm.size(); ++pe) {
            auto begin = input.begin() + pe * elements_per_pe;
            auto end   = begin + elements_per_pe;
            std::fill(begin, end, pe);
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
    auto const result = comm.scatter(send_buf(input), recv_count(1)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
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
    int const    root = comm.size() - 1; // use last PE as root

    std::vector<int> input(static_cast<std::size_t>(comm.size()));
    if (comm.rank() == root) {
        std::iota(input.begin(), input.end(), 0);
    }

    auto const result = comm.scatter(send_buf(input), kamping::root(root)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatter_with_nonzero_root_comm) {
    Communicator dummy_comm;
    int const    root = dummy_comm.size() - 1; // use last PE as root
    Communicator comm(MPI_COMM_WORLD, root);

    std::vector<int> input(static_cast<std::size_t>(comm.size()));
    if (comm.rank() == root) {
        std::iota(input.begin(), input.end(), 0);
    }

    auto const result = comm.scatter(send_buf(input)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatterv_with_one_element_per_pe) {
    Communicator comm;

    auto const             input = create_input_vector_on_root(comm, 1);
    std::vector<int> const counts(static_cast<std::size_t>(comm.size()), 1);
    std::vector<int>       displs(static_cast<std::size_t>(comm.size()));
    std::iota(displs.begin(), displs.end(), 0);

    auto const result = comm.scatterv(send_buf(input), send_counts(counts), send_displs(displs)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScatterTest, scatterv_with_one_element_per_pe_deduce_displs) {
    Communicator comm;

    auto const             input = create_input_vector_on_root(comm, 1);
    std::vector<int> const counts(static_cast<std::size_t>(comm.size()), 1);

    auto const result = comm.scatterv(send_buf(input), send_counts(counts)).extract_recv_buffer();
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

// 1 element on PE 0, 2 elements on PE 1, 3 elements on PE 2, ...
std::vector<int> create_triangle_input_vector_on_root(Communicator const& comm) {
    std::vector<int> input;

    for (int pe = 0; pe < comm.size(); ++pe) {
        for (int date = 0; date < pe + 1; ++date) {
            input.push_back(date);
        }
    }

    return input;
}

TEST(ScatterTest, scatterv_with_unequal_number_of_elements_per_pe) {
    Communicator comm;

    auto const input = create_triangle_input_vector_on_root(comm);

    std::vector<int> counts(static_cast<std::size_t>(comm.size()));
    std::iota(counts.begin(), counts.end(), 1);

    std::vector<int> displs(static_cast<std::size_t>(comm.size()));
    for (std::size_t pe = 1; pe < static_cast<std::size_t>(comm.size()); ++pe) {
        displs[pe] = displs[pe - 1] + static_cast<int>(pe);
    }

    auto const result = comm.scatterv(send_buf(input), send_counts(counts), send_displs(displs)).extract_recv_buffer();

    ASSERT_EQ(result.size(), comm.rank() + 1);

    std::vector<int> expected_result(static_cast<std::size_t>(comm.rank() + 1));
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(result, expected_result);
}

TEST(ScatterTest, scatterv_with_unequal_number_of_elements_per_pe_deduce_displs) {
    Communicator comm;

    auto const input = create_triangle_input_vector_on_root(comm);

    std::vector<int> counts(static_cast<std::size_t>(comm.size()));
    std::iota(counts.begin(), counts.end(), 1);

    auto const result = comm.scatterv(send_buf(input), send_counts(counts)).extract_recv_buffer();

    ASSERT_EQ(result.size(), comm.rank() + 1);

    std::vector<int> expected_result(static_cast<std::size_t>(comm.rank() + 1));
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(result, expected_result);
}
