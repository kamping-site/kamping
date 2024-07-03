// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "test_assertions.hpp"

#include "gmock/gmock.h"
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/comm_helper/num_numa_nodes.hpp"
#include "kamping/communicator.hpp"
#include "kamping/distributed_graph_communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(CommunicationGraphTest, empty) {
    uint8_t u8val = 200;
    EXPECT_TRUE(in_range<uint8_t>(u8val));
    EXPECT_TRUE(in_range<uint16_t>(u8val));
    EXPECT_TRUE(in_range<uint32_t>(u8val));
    EXPECT_TRUE(in_range<uint64_t>(u8val));
    EXPECT_FALSE(in_range<int8_t>(u8val));
    EXPECT_TRUE(in_range<int16_t>(u8val));
    EXPECT_TRUE(in_range<int32_t>(u8val));
    EXPECT_TRUE(in_range<int64_t>(u8val));
    u8val = 10;
    EXPECT_TRUE(in_range<int8_t>(u8val));

    auto intMax = std::numeric_limits<int>::max();
    EXPECT_TRUE(in_range<long int>(intMax));
    EXPECT_TRUE(in_range<uintmax_t>(intMax));
    EXPECT_TRUE(in_range<intmax_t>(intMax));

    auto intNeg = -1;
    EXPECT_TRUE(in_range<long int>(intNeg));
    EXPECT_FALSE(in_range<uintmax_t>(intNeg));
    EXPECT_TRUE(in_range<intmax_t>(intNeg));
    EXPECT_FALSE(in_range<size_t>(intNeg));
    EXPECT_TRUE(in_range<short int>(intNeg));

    size_t sizeT = 10000;
    EXPECT_TRUE(in_range<int>(sizeT));
    sizeT = std::numeric_limits<size_t>::max() - 1000;
    EXPECT_FALSE(in_range<int>(sizeT));
    EXPECT_TRUE(in_range<uintmax_t>(sizeT));

    unsigned long a = 16;
    EXPECT_TRUE(in_range<unsigned char>(a));

    // Cast large values into narrower types.
    EXPECT_FALSE(in_range<uint8_t>(std::numeric_limits<uint16_t>::max()));
    EXPECT_FALSE(in_range<uint16_t>(std::numeric_limits<uint32_t>::max() - 1000));
    EXPECT_FALSE(in_range<uint32_t>(std::numeric_limits<uint64_t>::max() - 133742));

    EXPECT_FALSE(in_range<int8_t>(std::numeric_limits<int16_t>::max()));
    EXPECT_FALSE(in_range<int8_t>(std::numeric_limits<int16_t>::min()));
    EXPECT_FALSE(in_range<int16_t>(std::numeric_limits<int32_t>::max()));
    EXPECT_FALSE(in_range<int16_t>(std::numeric_limits<int32_t>::min()));
    EXPECT_FALSE(in_range<int32_t>(std::numeric_limits<int64_t>::max()));
    EXPECT_FALSE(in_range<int32_t>(std::numeric_limits<int64_t>::min()));
}
