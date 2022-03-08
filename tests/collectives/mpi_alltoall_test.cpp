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
#include "kamping/kassert.hpp"
#include <algorithm>
#include <bits/stdint-uintn.h>
#undef KAMPING_ASSERTION_LEVEL
#define KAMPING_ASSERTION_LEVEL kamping::assert::normal
#ifndef KAMPING_EXCEPTION_MODE
    #define KAMPING_EXCEPTION_MODE
#endif // KAMPING_EXCEPTION_MODE

#include <gtest/gtest.h>
#include <mpi.h>
#include <numeric>

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

TEST(AlltoallTest, alltoall_custom_type) {
    Communicator comm;

    struct CustomType {
        int  sendingRank;
        int  receivingRank;
        bool operator==(const CustomType& other) const {
            return sendingRank == other.sendingRank && receivingRank == other.receivingRank;
        }
    };

    std::vector<CustomType> input(asserting_cast<size_t>(comm.size()));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = {comm.rank(), asserting_cast<int>(i)};
    }

    auto result = comm.alltoall(send_buf(input)).extract_recv_buffer();

    EXPECT_EQ(result.size(), comm.size());

    std::vector<CustomType> expected_result(asserting_cast<size_t>(comm.size()));
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

// TEST(AlltoallTest, alltoall_small_recv_type) {
//     Communicator comm;

//     std::vector<uint32_t> input(asserting_cast<size_t>(comm.size()));
//     std::iota(input.begin(), input.end(), 0);

//     std::vector<uint16_t> result;
//     comm.alltoall(send_buf(input), recv_buf(result));

//     EXPECT_EQ(result.size(), comm.size() * 2);

//     std::vector<uint16_t> expected_result(asserting_cast<size_t>(comm.size()) * 2);
//     for (size_t i = 0; i < expected_result.size(); ++i) {
//         // Assuming little-endian
//         if (i % 2 == 0) {
//             expected_result[i] = comm.rank() & 0x00ff;
//         } else {
//             expected_result[i] = (comm.rank() & 0xff00) >> 16;
//         }
//     }
//     EXPECT_EQ(result, expected_result);
// }

// TEST(AlltoallTest, alltoall_big_recv_type) {
//     Communicator comm;

//     std::vector<uint16_t> input(asserting_cast<size_t>(comm.size() * 2));
//     std::iota(input.begin(), input.end(), 0);
//     std::transform(
//         input.begin(), input.end(), input.begin(), [](const uint16_t element) -> uint16_t { return element / 2; });

//     std::vector<uint32_t> result;
//     comm.alltoall(send_buf(input), recv_buf(result));

//     EXPECT_EQ(result.size(), comm.size());

//     uint32_t expectedValue =
//         (asserting_cast<uint32_t>(comm.rank()) << 16) | (asserting_cast<uint32_t>(comm.rank()) & 0x00ff);
//     std::vector<uint32_t> expected_result(asserting_cast<size_t>(comm.size()), expectedValue);
//     EXPECT_EQ(result, expected_result);
// }

// TEST(AlltoallTest, alltoall_nondivisible_types) {
//     struct unsigned_24_bit_type {
//         uint8_t first;
//         uint8_t second;
//         uint8_t third;

//         bool operator==(uint32_t other) {
//             // assuming little endian
//             return (other & 0x000f) == first && ((other & 0x00f0) >> 8) == second && ((other & 0x0f00) >> 16) ==
//             third
//                    && (other & 0xf0000) == 0;
//         }
//     };
//     static_assert(sizeof(unsigned_24_bit_type) == 3, "unsigned_24_bit_type it not 3 bytes wide");

//     Communicator comm;

//     EXPECT_EQ(result, expected_result);
// }
