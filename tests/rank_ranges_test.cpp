// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <gtest/gtest.h>

#include "kamping/rank_ranges.hpp"

using namespace ::kamping;

TEST(RankRangesTest, construction_from_empty_c_style_array) {
    RankRanges rank_ranges(nullptr, 0);
    EXPECT_EQ(rank_ranges.size(), 0);
    EXPECT_EQ(rank_ranges.get(), nullptr);
    EXPECT_FALSE(rank_ranges.contains(0));
    EXPECT_FALSE(rank_ranges.contains(1));
}

TEST(RankRangesTest, construction_from_c_style_array) {
    int        rank_range_array[1][3] = {{1, 1, 1}};
    RankRanges rank_ranges(rank_range_array, 1);
    EXPECT_EQ(rank_ranges.size(), 1);
    EXPECT_EQ(rank_ranges.get(), rank_range_array);
    EXPECT_FALSE(rank_ranges.contains(0));
    EXPECT_TRUE(rank_ranges.contains(1));
    EXPECT_FALSE(rank_ranges.contains(2));
}

TEST(RankRangesTest, construction_from_c_style_array_multiple_ranges) {
    int        rank_range_array[2][3] = {{1, 1, 1}, {2, 6, 2}};
    RankRanges rank_ranges(rank_range_array, 2);
    EXPECT_EQ(rank_ranges.size(), 2);
    EXPECT_EQ(rank_ranges.get(), rank_range_array);
    EXPECT_FALSE(rank_ranges.contains(0));
    EXPECT_TRUE(rank_ranges.contains(1));
    EXPECT_TRUE(rank_ranges.contains(2));
    EXPECT_FALSE(rank_ranges.contains(3));
    EXPECT_TRUE(rank_ranges.contains(4));
    EXPECT_FALSE(rank_ranges.contains(5));
    EXPECT_TRUE(rank_ranges.contains(6));
    EXPECT_FALSE(rank_ranges.contains(7));
}

TEST(RankRangesTest, construction_from_empty_vector) {
    std::vector<RankRange> rank_range{};
    RankRanges             rank_ranges(rank_range);
    EXPECT_EQ(rank_ranges.size(), 0);
    EXPECT_FALSE(rank_ranges.contains(0));
    EXPECT_FALSE(rank_ranges.contains(1));
}
TEST(RankRangesTest, construction_from_vector) {
    RankRange              rank_range{1, 1, 1};
    std::vector<RankRange> rank_ranges_container{rank_range};
    RankRanges             rank_ranges(rank_ranges_container);
    EXPECT_EQ(rank_ranges.size(), 1);
    for (std::size_t i = 0; i < rank_ranges.size(); ++i) {
        EXPECT_EQ(rank_ranges.get()[i][0], rank_ranges_container[i].first);
        EXPECT_EQ(rank_ranges.get()[i][1], rank_ranges_container[i].last);
        EXPECT_EQ(rank_ranges.get()[i][2], rank_ranges_container[i].stride);
    }
    EXPECT_FALSE(rank_ranges.contains(0));
    EXPECT_TRUE(rank_ranges.contains(1));
    EXPECT_FALSE(rank_ranges.contains(2));
}

TEST(RankRangesTest, construction_from_vector_with_multiple_ranges) {
    RankRange              rank_range1{1, 1, 1};
    RankRange              rank_range2{2, 6, 2};
    std::vector<RankRange> rank_ranges_container{rank_range1, rank_range2};
    RankRanges             rank_ranges(rank_ranges_container);
    EXPECT_EQ(rank_ranges.size(), 2);
    for (std::size_t i = 0; i < rank_ranges.size(); ++i) {
        EXPECT_EQ(rank_ranges.get()[i][0], rank_ranges_container[i].first);
        EXPECT_EQ(rank_ranges.get()[i][1], rank_ranges_container[i].last);
        EXPECT_EQ(rank_ranges.get()[i][2], rank_ranges_container[i].stride);
    }
    EXPECT_FALSE(rank_ranges.contains(0));
    EXPECT_TRUE(rank_ranges.contains(1));
    EXPECT_TRUE(rank_ranges.contains(2));
    EXPECT_FALSE(rank_ranges.contains(3));
    EXPECT_TRUE(rank_ranges.contains(4));
    EXPECT_FALSE(rank_ranges.contains(5));
    EXPECT_TRUE(rank_ranges.contains(6));
    EXPECT_FALSE(rank_ranges.contains(7));
}
