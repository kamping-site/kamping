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

#include "../test_assertions.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/plugin/sort.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::plugin;

TEST(SortTest, sort_same_number_elements) {
    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_int_distribution<int32_t> dist;

    Communicator<std::vector, plugin::SampleSort> comm;
    size_t const                                  local_size = 2'000;
    std::vector<int32_t>                          local_data;
    for (size_t i = 0; i < local_size; ++i) {
        local_data.push_back(dist(gen));
    }

    auto original_data = local_data;

    comm.sort(local_data);
    EXPECT_TRUE(std::is_sorted(local_data.begin(), local_data.end()));

    std::array<int32_t, 2> borders = {local_data.front(), local_data.back()};

    auto all_borders = comm.allgather(send_buf(borders));
    EXPECT_TRUE(std::is_sorted(all_borders.begin(), all_borders.end()));

    auto total_expected_size = comm.allreduce_single(send_buf(local_size), op(ops::plus<>()));
    auto total_size          = comm.allreduce_single(send_buf(local_data.size()), op(ops::plus<>()));
    EXPECT_EQ(total_size, total_expected_size);

    auto all_sorted_data   = comm.gatherv(send_buf(local_data));
    auto all_original_data = comm.gatherv(send_buf(original_data));
    std::sort(all_original_data.begin(), all_original_data.end());
    ASSERT_EQ(all_sorted_data.size(), all_original_data.size());
    EXPECT_EQ(all_sorted_data, all_original_data);
}

TEST(SortTest, sort_same_number_elements_output_iterator) {
    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_int_distribution<int32_t> dist;

    Communicator<std::vector, plugin::SampleSort> comm;
    size_t const                                  local_size = 2'000;
    std::vector<int32_t>                          local_data;
    for (size_t i = 0; i < local_size; ++i) {
        local_data.push_back(dist(gen));
    }

    auto                 original_data = local_data;
    std::vector<int32_t> result;
    comm.sort(local_data.begin(), local_data.end(), std::back_inserter(result));
    EXPECT_TRUE(std::is_sorted(result.begin(), result.end()));

    std::array<int32_t, 2> borders = {result.front(), result.back()};

    auto all_borders = comm.allgather(send_buf(borders));
    EXPECT_TRUE(std::is_sorted(all_borders.begin(), all_borders.end()));

    auto total_expected_size = comm.allreduce_single(send_buf(local_size), op(ops::plus<>()));
    auto total_size          = comm.allreduce_single(send_buf(result.size()), op(ops::plus<>()));
    EXPECT_EQ(total_size, total_expected_size);

    auto all_sorted_data   = comm.gatherv(send_buf(result));
    auto all_original_data = comm.gatherv(send_buf(original_data));
    std::sort(all_original_data.begin(), all_original_data.end());
    ASSERT_EQ(all_sorted_data.size(), all_original_data.size());
    EXPECT_EQ(all_sorted_data, all_original_data);
}

TEST(SortTest, sort_different_number_elements) {
    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_int_distribution<int32_t> dist;

    Communicator<std::vector, plugin::SampleSort> comm;
    size_t const                                  local_size = 2'000 * comm.rank();
    std::vector<int32_t>                          local_data;
    for (size_t i = 0; i < local_size; ++i) {
        local_data.push_back(dist(gen));
    }

    auto original_data = local_data;

    comm.sort(local_data);

    if (local_data.size() > 0) {
        EXPECT_TRUE(std::is_sorted(local_data.begin(), local_data.end()));

        std::array<int32_t, 2> borders = {local_data.front(), local_data.back()};

        auto all_borders = comm.allgather(send_buf(borders));
        EXPECT_TRUE(std::is_sorted(all_borders.begin(), all_borders.end()));

        auto total_expected_size = comm.allreduce_single(send_buf(local_size), op(ops::plus<>()));
        auto total_size          = comm.allreduce_single(send_buf(local_data.size()), op(ops::plus<>()));
        EXPECT_EQ(total_size, total_expected_size);

        auto all_sorted_data   = comm.gatherv(send_buf(local_data));
        auto all_original_data = comm.gatherv(send_buf(original_data));
        std::sort(all_original_data.begin(), all_original_data.end());
        ASSERT_EQ(all_sorted_data.size(), all_original_data.size());
        EXPECT_EQ(all_sorted_data, all_original_data);
    }
}

TEST(SortTest, sort_non_default_comparator) {
    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_int_distribution<int32_t> dist;

    Communicator<std::vector, plugin::SampleSort> comm;
    size_t const                                  local_size = 2'000;
    std::vector<int32_t>                          local_data;
    for (size_t i = 0; i < local_size; ++i) {
        local_data.push_back(dist(gen));
    }

    auto original_data = local_data;

    comm.sort(local_data, std::greater<int32_t>());
    EXPECT_TRUE(std::is_sorted(local_data.begin(), local_data.end(), std::greater<int32_t>()));

    std::array<int32_t, 2> borders = {local_data.front(), local_data.back()};

    auto all_borders = comm.allgather(send_buf(borders));
    EXPECT_TRUE(std::is_sorted(all_borders.begin(), all_borders.end(), std::greater<int32_t>()));

    auto total_expected_size = comm.allreduce_single(send_buf(local_size), op(ops::plus<>()));
    auto total_size          = comm.allreduce_single(send_buf(local_data.size()), op(ops::plus<>()));
    EXPECT_EQ(total_size, total_expected_size);

    auto all_sorted_data   = comm.gatherv(send_buf(local_data));
    auto all_original_data = comm.gatherv(send_buf(original_data));
    std::sort(all_original_data.begin(), all_original_data.end(), std::greater<int32_t>());
    ASSERT_EQ(all_sorted_data.size(), all_original_data.size());
    EXPECT_EQ(all_sorted_data, all_original_data);
}

TEST(SortTest, sort_custom_type) {
    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_int_distribution<int32_t> dist;

    struct MyStruct {
        int32_t x;
        int32_t y;
        int32_t z;

        MyStruct() = default;

        MyStruct(int32_t _x, int32_t _y, int32_t _z) : x(_x), y(_y), z(_z) {}

        bool operator==(MyStruct other) const {
            return std::make_tuple(x, y, z) == std::make_tuple(other.x, other.y, other.z);
        }

        bool operator<(MyStruct other) const {
            return std::make_tuple(x, y, z) < std::make_tuple(other.x, other.y, other.z);
        }
    };

    Communicator<std::vector, plugin::SampleSort> comm;
    size_t const                                  local_size = 2'000;
    std::vector<MyStruct>                         local_data;
    for (size_t i = 0; i < local_size; ++i) {
        local_data.emplace_back(dist(gen), dist(gen), dist(gen));
    }

    auto original_data = local_data;

    comm.sort(local_data);
    EXPECT_TRUE(std::is_sorted(local_data.begin(), local_data.end()));

    std::array<MyStruct, 2> borders = {local_data.front(), local_data.back()};

    auto all_borders = comm.allgather(send_buf(borders));
    EXPECT_TRUE(std::is_sorted(all_borders.begin(), all_borders.end()));

    auto total_expected_size = comm.allreduce_single(send_buf(local_size), op(ops::plus<>()));
    auto total_size          = comm.allreduce_single(send_buf(local_data.size()), op(ops::plus<>()));
    EXPECT_EQ(total_size, total_expected_size);

    auto all_sorted_data   = comm.gatherv(send_buf(local_data));
    auto all_original_data = comm.gatherv(send_buf(original_data));
    std::sort(all_original_data.begin(), all_original_data.end());
    ASSERT_EQ(all_sorted_data.size(), all_original_data.size());
    EXPECT_EQ(all_sorted_data, all_original_data);
}
