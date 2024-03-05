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

#include <cstddef>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kamping/communicator.hpp"
#include "kamping/plugin/sort.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::plugin;

TEST(SortTest, sort_same_number_elements) {
    Communicator<std::vector, plugin::SampleSort> comm;

    std::vector<int32_t> local_data;
    for (size_t i = 0; i < 10'000; ++i) {
        local_data.push_back(rand());
    }

    comm.sort(local_data.begin(), local_data.end());
}
