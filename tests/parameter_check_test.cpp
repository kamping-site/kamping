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

#include <gtest/gtest.h>
#include <vector>

#include "helpers_for_testing.hpp"
#include "kamping/parameter_check.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace {
template <typename... Args>
void test_empty_arguments(Args&&...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS());
}

template <typename... Args>
void test_required_send_buf(Args&&...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS());
}

template <typename... Args>
void test_required_send_buf_optional_recv_buf(Args&&...) {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf), KAMPING_OPTIONAL_PARAMETERS(recv_buf));
}

template <typename... Args>
void test_optional_recv_buf(Args&&...) {
    // KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(recv_buf));
    //
    // generates the following broken code:
    do {
        static_assert(
            kamping::internal::all_parameters_are_rvalues<Args...>,
            "All parameters have to be passed in as rvalue references, meaning that you must not hold a variable "
            "returned by the named parameter helper functions like recv_buf().");
        ;
        using required_parameters_types = typename kamping::internal::parameter_types_to_integral_constants<>::type;
        using optional_parameters_types = typename kamping::internal::parameter_types_to_integral_constants<
            kamping::internal::ParameterType::recv_buf>::type;
        using parameter_types = typename kamping::internal::parameters_to_integral_constant<Args...>::type;
        static_assert(
            kamping::internal::has_no_unused_parameters<
                required_parameters_types, optional_parameters_types, Args...>::assertion,
            "There are unsupported parameters, only support required "
            "parameters "
            ""
            " and optional parameters "
            ",recv_buf");
        static_assert(kamping::internal::all_unique_v<parameter_types>, "There are duplicate parameter types.");
    } while (false);
}
} // namespace

TEST(ParameterCheckTest, check_empty) {
    test_empty_arguments();
}

TEST(ParameterCheckTest, check_required) {
    std::vector<int> v;
    test_required_send_buf(kamping::send_buf(v));
}

TEST(ParameterCheckTest, check_required_and_optional) {
    std::vector<int> v;
    test_required_send_buf_optional_recv_buf(kamping::send_buf(v));
    test_required_send_buf_optional_recv_buf(kamping::send_buf(v), kamping::recv_buf(v));
}

TEST(ParameterCheckTest, check_optional) {
    std::vector<int> v;
    test_optional_recv_buf();
    test_optional_recv_buf(kamping::recv_buf(v));
}

namespace {
// @brief This dummy resembles the interface of a collective operation, so we can simulate the check for rvalue
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
