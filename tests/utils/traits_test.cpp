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

#include <list>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <kamping/utils/traits.hpp>

TEST(TraitsTest, is_range) {
    EXPECT_TRUE(kamping::is_range_v<std::vector<int>>);
    EXPECT_FALSE(kamping::is_range_v<int>);
}

TEST(TraitsTest, is_contiguous_sized_range) {
    EXPECT_TRUE(kamping::is_contiguous_sized_range_v<std::vector<int>>);
    EXPECT_FALSE(kamping::is_contiguous_sized_range_v<std::list<int>>);
    EXPECT_FALSE(kamping::is_contiguous_sized_range_v<int>);
}

TEST(TraitsTest, is_pair_like) {
    EXPECT_TRUE((kamping::is_pair_like_v<std::pair<int, int>>));
    EXPECT_TRUE((kamping::is_pair_like_v<std::tuple<int, int>>));
    EXPECT_FALSE((kamping::is_pair_like_v<std::tuple<int, int, int>>));
    EXPECT_FALSE(kamping::is_pair_like_v<int>);
}

TEST(TraitsTest, is_destination_buffer_pair) {
    EXPECT_TRUE((kamping::is_destination_buffer_pair_v<std::pair<int, std::vector<int>>>));
    EXPECT_FALSE((kamping::is_destination_buffer_pair_v<std::vector<int>>));
    EXPECT_FALSE((kamping::is_destination_buffer_pair_v<std::pair<std::string, std::vector<int>>>));
}

TEST(TraitsTest, is_sparse_send_buffer) {
    EXPECT_TRUE((kamping::is_sparse_send_buffer_v<std::unordered_map<int, std::vector<int>>>));
    EXPECT_TRUE((kamping::is_sparse_send_buffer_v<std::vector<std::pair<int, std::vector<int>>>>));
    EXPECT_FALSE(kamping::is_sparse_send_buffer_v<std::vector<int>>);
    EXPECT_FALSE(kamping::is_sparse_send_buffer_v<std::vector<std::vector<int>>>);
}
