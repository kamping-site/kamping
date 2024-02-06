// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(AssertionHelpersTests, is_same_on_all_ranks) {
    Communicator comm;

    // All ranks have the same value.
    size_t value = 0;
    EXPECT_TRUE(comm.is_same_on_all_ranks(value));

    // PE with rank 0 has a different value.
    if (comm.rank() == 0) {
        value = 1;
    }
    if (comm.size() > 1) {
        EXPECT_FALSE(comm.is_same_on_all_ranks(value));
    } else {
        EXPECT_TRUE(comm.is_same_on_all_ranks(value));
    }

    // Try different data types.
    int           value_int         = 0;
    unsigned long value_ulint       = 10;
    short const   value_const_short = 0;
    bool          value_bool        = false;
    float         value_float       = 0.0;
    double        value_double      = 0.0;
    char          value_char        = 'a';

    enum ValueEnum { a, b, c };
    enum class ValueEnumClass { a, b, c };

    struct ValueStruct {
        int a;
        int b;

        bool operator==(ValueStruct const& that) const {
            return this->a == that.a && this->b == that.b;
        }
    };
    ValueStruct value_struct = {0, 0};

    EXPECT_TRUE(comm.is_same_on_all_ranks(value_int));
    EXPECT_TRUE(comm.is_same_on_all_ranks(value_ulint));
    EXPECT_TRUE(comm.is_same_on_all_ranks(value_const_short));
    EXPECT_TRUE(comm.is_same_on_all_ranks(value_bool));
    EXPECT_TRUE(comm.is_same_on_all_ranks(value_float));
    EXPECT_TRUE(comm.is_same_on_all_ranks(value_double));
    EXPECT_TRUE(comm.is_same_on_all_ranks(value_char));
    EXPECT_TRUE(comm.is_same_on_all_ranks(ValueEnum::a));
    EXPECT_TRUE(comm.is_same_on_all_ranks(ValueEnumClass::b));
    EXPECT_TRUE(comm.is_same_on_all_ranks(value_struct));

    if (comm.rank() == 0) {
        value_int      = 1;
        value_ulint    = 1;
        value_bool     = true;
        value_float    = 1.0;
        value_double   = -1.0;
        value_char     = 'b';
        value_struct.a = 1;
    }

    if (comm.size() > 1) {
        EXPECT_FALSE(comm.is_same_on_all_ranks(value_int));
        EXPECT_FALSE(comm.is_same_on_all_ranks(value_ulint));
        EXPECT_FALSE(comm.is_same_on_all_ranks(value_bool));
        EXPECT_FALSE(comm.is_same_on_all_ranks(value_float));
        EXPECT_FALSE(comm.is_same_on_all_ranks(value_double));
        EXPECT_FALSE(comm.is_same_on_all_ranks(value_char));
        EXPECT_FALSE(comm.is_same_on_all_ranks(value_struct));
    } else {
        EXPECT_TRUE(comm.is_same_on_all_ranks(value_int));
        EXPECT_TRUE(comm.is_same_on_all_ranks(value_ulint));
        EXPECT_TRUE(comm.is_same_on_all_ranks(value_bool));
        EXPECT_TRUE(comm.is_same_on_all_ranks(value_float));
        EXPECT_TRUE(comm.is_same_on_all_ranks(value_double));
        EXPECT_TRUE(comm.is_same_on_all_ranks(value_char));
        EXPECT_TRUE(comm.is_same_on_all_ranks(value_struct));
    }

    if (comm.size() > 1) {
        // Compare non-equal const-values.
        if (comm.is_root()) {
            short const value_const_short_2 = 42;
            EXPECT_FALSE(comm.is_same_on_all_ranks(value_const_short_2));
        } else {
            EXPECT_FALSE(comm.is_same_on_all_ranks(value_const_short));
        }

        // Compare non-equal enums.
        if (comm.is_root()) {
            EXPECT_FALSE(comm.is_same_on_all_ranks(ValueEnum::a));
            EXPECT_FALSE(comm.is_same_on_all_ranks(ValueEnumClass::a));
        } else {
            EXPECT_FALSE(comm.is_same_on_all_ranks(ValueEnum::b));
            EXPECT_FALSE(comm.is_same_on_all_ranks(ValueEnumClass::b));
        }
    }
}
