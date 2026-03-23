// This file is part of KaMPIng.
//
// Copyright 2021-2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <gtest/gtest.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/kassert/kassert.hpp"

//
// Makros to test for failed KAMPING_ASSERT() statements by using a death test. This is usually only a last resort, when
// we want to test KAMPING_ASSERTs in places which are not allowed to throw exceptions.
//
// See test_assertions.hpp for the exception overriding version.
//
// Note that these macros could already be defined if we included the header that turns assertions into exceptions. In
// this case, we keep the current definition.
//

#ifndef EXPECT_KAMPING_ASSERT_FAILS_WITH_DEATH
    // EXPECT that a KAMPING_ASSERT assertion failed and that the error message contains a certain failure_message.
    #define EXPECT_KAMPING_ASSERT_FAILS_WITH_DEATH(code, failure_message) \
        EXPECT_EXIT({ code; }, ::testing::KilledBySignal(SIGABRT), failure_message);
#endif

#ifndef ASSERT_KAMPING_ASSERT_FAILS_WITH_DEATH
    // ASSERT that a KAMPING_ASSERT assertion failed and that the error message contains a certain failure_message.
    #define ASSERT_KAMPING_ASSERT_FAILS_WITH_DEATH(code, failure_message) \
        ASSERT_EXIT({ code; }, ::testing::KilledBySignal(SIGABRT), failure_message);
#endif
