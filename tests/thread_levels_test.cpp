// This file is part of KaMPIng.
//
// Copyright 2022-2026 The KaMPIng Authors
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

#include "kamping/thread_levels.hpp"

TEST(ThreadLevelsTest, levels_are_monotonic) {
    EXPECT_LT(kamping::ThreadLevel::single, kamping::ThreadLevel::funneled);
    EXPECT_LT(kamping::ThreadLevel::funneled, kamping::ThreadLevel::serialized);
    EXPECT_LT(kamping::ThreadLevel::serialized, kamping::ThreadLevel::multiple);
}
