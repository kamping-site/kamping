// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

// overwrite build options and set assertion level to normal, enable exceptions
#undef KAMPING_ASSERTION_LEVEL
#define KAMPING_ASSERTION_LEVEL kamping::assert::normal
#ifndef KAMPING_EXCEPTION_MODE
    #define KAMPING_EXCEPTION_MODE
#endif // KAMPING_EXCEPTION_MODE

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

#include "../helpers_for_testing.hpp"
#include "kamping/communicator.hpp"


using namespace ::kamping;
using namespace ::testing;

TEST(AlltoallTest, alltoall_single_element_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input(asserting_cast<size_t>(comm.size()));
    std::iota(input.begin(), input.end(), 0);

    auto result = comm.alltoall(send_buf(input)).extract_recv_buffer();

    EXPECT_EQ(result.size(), comm.size());

    std::vector<int> expected_result(asserting_cast<size_t>(comm.size()), comm.rank());
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, alltoall_single_element_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input(asserting_cast<size_t>(comm.size()), comm.rank());

    std::vector<int> result;
    comm.alltoall(send_buf(input), recv_buf(result));

    EXPECT_EQ(result.size(), comm.size());

    std::vector<int> expected_result(asserting_cast<size_t>(comm.size()));
    std::iota(expected_result.begin(), expected_result.end(), 0);
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, alltoall_multiple_elements) {
    Communicator comm;

    const int num_elements_per_processor_pair = 4;

    std::vector<int> input(asserting_cast<size_t>(comm.size() * num_elements_per_processor_pair));
    std::iota(input.begin(), input.end(), 0);
    std::transform(input.begin(), input.end(), input.begin(), [](const int element) -> int {
        return element / num_elements_per_processor_pair;
    });

    std::vector<int> result;
    comm.alltoall(send_buf(input), recv_buf(result));

    EXPECT_EQ(result.size(), comm.size() * num_elements_per_processor_pair);

    std::vector<int> expected_result(
        asserting_cast<size_t>(comm.size() * num_elements_per_processor_pair), comm.rank());
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, alltoall_custom_type_custom_container) {
    Communicator comm;

    struct CustomType {
        int  sendingRank;
        int  receivingRank;
        bool operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    OwnContainer<CustomType> input(asserting_cast<size_t>(comm.size()));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = {comm.rank(), asserting_cast<int>(i)};
    }

    auto result =
        comm.alltoall(send_buf(input), recv_buf(NewContainer<OwnContainer<CustomType>>{})).extract_recv_buffer();

    EXPECT_EQ(result.size(), comm.size());

    OwnContainer<CustomType> expected_result(asserting_cast<size_t>(comm.size()));
    for (size_t i = 0; i < expected_result.size(); ++i) {
        expected_result[i] = {asserting_cast<int>(i), comm.rank()};
    }
    EXPECT_EQ(result, expected_result);
}

TEST(AlltoallTest, alltoall_mismatched_types) {
    Communicator comm;

    const int num_elements_per_processor_pair = 4;

    std::vector<int> input(asserting_cast<size_t>(comm.size() * num_elements_per_processor_pair));
    std::iota(input.begin(), input.end(), 0);
    std::transform(input.begin(), input.end(), input.begin(), [](const int element) -> int {
        return element / num_elements_per_processor_pair;
    });

    std::vector<float> result;

    EXPECT_THROW(comm.alltoall(send_buf(input), recv_buf(result)), KassertException);
}
