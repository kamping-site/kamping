// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <vector>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/parameter_check.hpp"
#include "kamping/parameter_factories.hpp"
#include "parameter_check_common.hpp"

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
    testing::test_require_many_parameters(
        send_buf(v), recv_buf(v), root(0), recv_count(0), recv_counts(v), send_counts(v));
}

namespace {
// This dummy resembles the interface of a collective operation, so we can simulate the check for rvalue
// parameters.
template <typename... Args>
bool dummy_collective_operation(Args&&... args [[maybe_unused]]) {
    return kamping::internal::all_parameters_are_rvalues<Args...>;
}
} // namespace

TEST(NamedParameterTest, all_parameters_are_rvalues) {
    using namespace kamping::internal;

    testing::Argument<ParameterType::send_buf> arg0{0};
    testing::Argument<ParameterType::recv_buf> arg1{1};
    {
        EXPECT_FALSE(dummy_collective_operation(arg0, arg1));
        EXPECT_FALSE(dummy_collective_operation(decltype(arg0){0}, arg1));
        EXPECT_FALSE(dummy_collective_operation(arg0, decltype(arg1){1}));
        EXPECT_TRUE(dummy_collective_operation(decltype(arg0){0}, decltype(arg1){1}));
    }
}
