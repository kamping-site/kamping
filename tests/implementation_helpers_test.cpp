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

#include <gmock/gmock.h>
#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>

#include "kamping/communicator.hpp"
#include "kamping/implementation_helpers.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/parameter_objects.hpp"

using namespace ::testing;
using namespace ::kamping;

TEST(ImplementationHelpersTest, is_valid_rank_in_comm) {
    kamping::Communicator comm;

    auto valid_value_rank_parameter = source(0);
    EXPECT_TRUE(kamping::internal::is_valid_rank_in_comm(valid_value_rank_parameter, comm, false, false));
    EXPECT_TRUE(kamping::internal::is_valid_rank_in_comm(valid_value_rank_parameter, comm, true, false));
    EXPECT_TRUE(kamping::internal::is_valid_rank_in_comm(valid_value_rank_parameter, comm, false, true));
    EXPECT_TRUE(kamping::internal::is_valid_rank_in_comm(valid_value_rank_parameter, comm, true, true));

    auto invalid_value_rank_parameter = source(comm.size());
    EXPECT_FALSE(kamping::internal::is_valid_rank_in_comm(invalid_value_rank_parameter, comm, false, false));
    EXPECT_FALSE(kamping::internal::is_valid_rank_in_comm(invalid_value_rank_parameter, comm, true, false));
    EXPECT_FALSE(kamping::internal::is_valid_rank_in_comm(invalid_value_rank_parameter, comm, false, true));
    EXPECT_FALSE(kamping::internal::is_valid_rank_in_comm(invalid_value_rank_parameter, comm, true, true));

    auto null_rank_parameter = source(rank::null);
    EXPECT_FALSE(kamping::internal::is_valid_rank_in_comm(null_rank_parameter, comm, false, false));
    EXPECT_TRUE(kamping::internal::is_valid_rank_in_comm(null_rank_parameter, comm, true, false));
    EXPECT_FALSE(kamping::internal::is_valid_rank_in_comm(null_rank_parameter, comm, false, true));
    EXPECT_TRUE(kamping::internal::is_valid_rank_in_comm(null_rank_parameter, comm, true, true));

    auto any_rank_parameter = source(rank::any);
    EXPECT_FALSE(kamping::internal::is_valid_rank_in_comm(any_rank_parameter, comm, false, false));
    EXPECT_FALSE(kamping::internal::is_valid_rank_in_comm(any_rank_parameter, comm, true, false));
    EXPECT_TRUE(kamping::internal::is_valid_rank_in_comm(any_rank_parameter, comm, false, true));
    EXPECT_TRUE(kamping::internal::is_valid_rank_in_comm(any_rank_parameter, comm, true, true));
}
