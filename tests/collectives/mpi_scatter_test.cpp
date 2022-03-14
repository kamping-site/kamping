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
