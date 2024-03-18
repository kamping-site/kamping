// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#include <algorithm>
#include <optional>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/communicator.hpp"
#include "kamping/measurements/internal/measurement_utils.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace kamping::measurements;
using namespace kamping::measurements::internal;

TEST(TimerUtilsTest, is_string_same_on_all_ranks_basics) {
    Communicator<>    comm;
    std::string const empty;
    std::string const non_empty("abc");
    EXPECT_TRUE(is_string_same_on_all_ranks(empty, comm));
    EXPECT_TRUE(is_string_same_on_all_ranks(non_empty, comm));
}

TEST(TimerUtilsTest, is_string_same_on_all_ranks) {
    Communicator<> comm;
    if (comm.size() <= 1) {
        return;
    }
    std::string str("abc");
    if (comm.rank() + 1 == comm.size()) {
        str = "cba";
    }
    EXPECT_FALSE(is_string_same_on_all_ranks(str, comm));
}
