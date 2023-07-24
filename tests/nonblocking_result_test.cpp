// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
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
#include <mpi.h>

#include "helpers_for_testing.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

using namespace kamping;

static bool test_succeed = false;

KAMPING_MAKE_HAS_MEMBER(wait)
KAMPING_MAKE_HAS_MEMBER(test)

int MPI_Wait(MPI_Request*, MPI_Status*) {
    return MPI_SUCCESS;
}

int MPI_Test(MPI_Request*, int* flag, MPI_Status*) {
    *flag = test_succeed;
    return MPI_SUCCESS;
}

class NonBlockingResultTest : public ::testing::Test {
    void SetUp() override {
        test_succeed = false;
    }
    void TearDown() override {
        test_succeed = false;
    }
};

TEST_F(NonBlockingResultTest, owning_request_and_result_types_match) {
    auto recv_buf_obj          = recv_buf(alloc_new<std::vector<int>>);
    using expected_result_type = MPIResult<
        internal::ResultCategoryNotUsed,
        decltype(recv_buf_obj),
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed>;
    auto request_obj = request();
    auto result      = kamping::make_nonblocking_result(std::move(recv_buf_obj), std::move(request_obj));

    EXPECT_TRUE(has_member_test_v<decltype(result)>);
    using test_return_type = decltype(result.test());
    EXPECT_TRUE((internal::is_specialization<test_return_type, std::optional>::value));
    EXPECT_TRUE((std::is_same_v<test_return_type::value_type, expected_result_type>));

    EXPECT_TRUE(has_member_wait_v<decltype(result)>);
    using wait_return_type = decltype(result.wait());
    EXPECT_TRUE((std::is_same_v<wait_return_type, expected_result_type>));
}

TEST_F(NonBlockingResultTest, owning_request_and_result_wait_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>);
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj   = request();
    auto result        = kamping::make_nonblocking_result(std::move(recv_buf_obj), std::move(request_obj));
    auto data          = result.wait().extract_recv_buffer();
    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data, expected_data);
    EXPECT_KASSERT_FAILS(result.extract(), "The result of this request has already been extracted.");
}

TEST_F(NonBlockingResultTest, owning_request_and_result_test_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>);
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj = request();
    auto result      = kamping::make_nonblocking_result(std::move(recv_buf_obj), std::move(request_obj));
    test_succeed     = false;
    EXPECT_FALSE(result.test().has_value());
    test_succeed = true;
    auto data    = result.test();
    EXPECT_TRUE(data.has_value());
    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data.value().extract_recv_buffer(), expected_data);
}

TEST_F(NonBlockingResultTest, owning_request_and_result_extract_works) {
    auto recv_buf_obj          = recv_buf(alloc_new<std::vector<int>>);
    using expected_result_type = MPIResult<
        internal::ResultCategoryNotUsed,
        decltype(recv_buf_obj),
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed>;
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj        = request();
    auto nonblocking_result = kamping::make_nonblocking_result(std::move(recv_buf_obj), std::move(request_obj));
    auto [req, result]      = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(req), Request>));
    EXPECT_TRUE((std::is_same_v<decltype(result), expected_result_type>));

    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(result.extract_recv_buffer(), expected_data);
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_types_match) {
    auto request_obj = request();
    auto result      = kamping::make_nonblocking_result(std::move(request_obj));
    EXPECT_TRUE(has_member_test_v<decltype(result)>);
    using test_return_type = decltype(result.test());
    EXPECT_TRUE((std::is_same_v<test_return_type, bool>));
    EXPECT_TRUE(has_member_wait_v<decltype(result)>);
    using wait_return_type = decltype(result.wait());
    EXPECT_TRUE((std::is_same_v<wait_return_type, void>));
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_test_works) {
    auto request_obj = request();
    auto result      = kamping::make_nonblocking_result(std::move(request_obj));
    test_succeed     = false;
    EXPECT_FALSE(result.test());
    test_succeed = true;
    EXPECT_TRUE(result.test());
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_extract_works) {
    using expected_result_type = MPIResult<
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed>;
    auto request_obj        = request();
    auto nonblocking_result = kamping::make_nonblocking_result(std::move(request_obj));
    auto [req, result]      = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(req), Request>));
    EXPECT_TRUE((std::is_same_v<decltype(result), expected_result_type>));

    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_types_match) {
    auto    recv_buf_obj = recv_buf(alloc_new<std::vector<int>>);
    Request req;
    auto    request_obj = request(req);
    auto    result      = kamping::make_nonblocking_result(std::move(recv_buf_obj), std::move(request_obj));
    EXPECT_FALSE(has_member_test_v<decltype(result)>)
        << "The result does not own the request, so test() should not be available.";
    EXPECT_FALSE(has_member_wait_v<decltype(result)>)
        << "The result does not own the request, so wait() should not be available.";
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_extract_works) {
    auto recv_buf_obj          = recv_buf(alloc_new<std::vector<int>>);
    using expected_result_type = MPIResult<
        internal::ResultCategoryNotUsed,
        decltype(recv_buf_obj),
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed,
        internal::ResultCategoryNotUsed>;
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    Request req;
    auto    request_obj        = request(req);
    auto    nonblocking_result = kamping::make_nonblocking_result(std::move(recv_buf_obj), std::move(request_obj));
    auto    result             = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(result), expected_result_type>));

    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(result.extract_recv_buffer(), expected_data);
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
}
