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

#include <vector>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameters.hpp"
#include "named_parameter_check_common.hpp"

TEST(ParameterCheckTest, check_empty) {
    testing::test_empty_arguments();
}

TEST(ParameterCheckTest, check_required) {
    std::vector<int> v;
    testing::test_required_send_buf(kamping::send_buf(v));
}

TEST(ParameterCheckTest, check_required_and_optional) {
    std::vector<int> v;
    testing::test_required_send_buf_optional_recv_buf(kamping::send_buf(v));
    testing::test_required_send_buf_optional_recv_buf(kamping::send_buf(v), kamping::recv_buf(v));
}

TEST(ParameterCheckTest, check_optional) {
    std::vector<int> v;
    testing::test_optional_recv_buf();
    testing::test_optional_recv_buf(kamping::recv_buf(v));
}

TEST(ParameterCheckTest, check_two_required_parameters) {
    using namespace kamping;
    std::vector<int> v;
    testing::test_required_send_recv_buf(send_buf(v), recv_buf(v));
}

TEST(ParameterCheckTest, check_two_optional_parameters) {
    using namespace kamping;
    std::vector<int> v;
    testing::test_optional_send_recv_buf(send_buf(v), recv_buf(v));
    testing::test_optional_send_recv_buf(send_buf(v));
    testing::test_optional_send_recv_buf(recv_buf(v));
    testing::test_optional_send_recv_buf();
}

TEST(ParameterCheckTest, check_many_required_parameters) {
    using namespace kamping;
    std::vector<int> v;
    testing::test_require_many_parameters(send_buf(v), recv_buf(v), root(0), recv_counts(v), send_counts(v));
}

TEST(NamedParameterTest, has_to_be_computed) {
    using namespace kamping::internal;

    std::vector<int> dummy_recv_counts;
    auto             recv_counts_in = kamping::recv_counts(dummy_recv_counts);
    EXPECT_FALSE(has_to_be_computed<decltype(recv_counts_in)>);

    auto recv_counts_out = kamping::recv_counts_out(kamping::alloc_new<std::vector<int>>);
    EXPECT_TRUE(has_to_be_computed<decltype(recv_counts_out)>);
}

TEST(NamedParameterTets, all_have_any_has_to_be_computed) {
    using namespace kamping::internal;

    std::vector<int> dummy;
    auto             recv_counts_given = kamping::recv_counts(dummy);
    auto             send_counts_given = kamping::send_counts(dummy);
    auto             recv_counts_empty = kamping::recv_counts_out(kamping::alloc_new<std::vector<int>>);
    auto             send_counts_empty = kamping::send_counts_out(kamping::alloc_new<std::vector<int>>);

    bool const all_positive = all_have_to_be_computed<decltype(recv_counts_empty), decltype(send_counts_empty)>;
    EXPECT_TRUE(all_positive);

    bool const all_negative = all_have_to_be_computed<decltype(recv_counts_given), decltype(recv_counts_empty)>;
    EXPECT_FALSE(all_negative);

    bool const any_positive =
        any_has_to_be_computed<decltype(recv_counts_given), decltype(recv_counts_empty), decltype(send_counts_empty)>;
    EXPECT_TRUE(any_positive);

    bool const any_negative = any_has_to_be_computed<decltype(recv_counts_given), decltype(send_counts_given)>;
    EXPECT_FALSE(any_negative);
}
