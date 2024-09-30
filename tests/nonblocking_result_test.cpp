// This file is part of KaMPIng.
//
// Copyright 2023-2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "test_assertions.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

using namespace kamping;

static bool   let_mpi_test_succeed = false;
static size_t num_wait_calls       = 0;
int const     TOUCHED_BY_MOCK_TAG  = 42;

KAMPING_MAKE_HAS_MEMBER(wait)
KAMPING_MAKE_HAS_MEMBER(test)

int MPI_Wait(MPI_Request*, MPI_Status* status) {
    // we have to do something useful here, because else clang wants us to make this function const, which fails the
    // build.
    if (status != MPI_STATUS_IGNORE) {
        status->MPI_TAG = TOUCHED_BY_MOCK_TAG;
    }
    num_wait_calls++;
    return MPI_SUCCESS;
}

int MPI_Test(MPI_Request*, int* flag, MPI_Status* status) {
    if (status != MPI_STATUS_IGNORE) {
        status->MPI_TAG = TOUCHED_BY_MOCK_TAG;
    }
    *flag = let_mpi_test_succeed;
    return MPI_SUCCESS;
}

class NonBlockingResultTest : public ::testing::Test {
    void SetUp() override {
        let_mpi_test_succeed = false;
        num_wait_calls       = 0;
    }
    void TearDown() override {
        let_mpi_test_succeed = false;
        num_wait_calls       = 0;
    }
};

TEST_F(NonBlockingResultTest, owning_request_and_result_types_match) {
    auto recv_buf_obj          = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    auto request_obj           = request();
    using RecvBufType          = decltype(recv_buf_obj);
    using RequestType          = decltype(request_obj);
    using expected_result_type = decltype(internal::make_mpi_result<std::tuple<RecvBufType, RequestType>>(
        std::move(recv_buf_obj),
        std::move(request_obj)
    ));
    auto buffer_on_heap        = move_buffer_to_heap(std::move(recv_buf_obj));
    auto result                = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        std::move(buffer_on_heap)
    );

    EXPECT_TRUE(has_member_test_v<decltype(result)>);
    EXPECT_TRUE(has_member_wait_v<decltype(result)>);
    {
        // ignore status -> return result only
        using test_return_type = decltype(result.test());
        EXPECT_TRUE((internal::is_specialization<test_return_type, std::optional>::value));
        EXPECT_TRUE((std::is_same_v<test_return_type::value_type, expected_result_type>));
    }
    {
        // also return status -> optional<pair<result, status>>
        using test_return_type = decltype(result.test(status_out()));
        EXPECT_TRUE((internal::is_specialization<test_return_type, std::optional>::value));
        EXPECT_TRUE((std::is_same_v<test_return_type::value_type::first_type, expected_result_type>));
        EXPECT_TRUE((std::is_same_v<test_return_type::value_type::second_type, Status>));
    }
    {
        // also return status, but as out parameter -> optional<result>
        Status status;
        using test_return_type = decltype(result.test(status_out(status)));
        EXPECT_TRUE((internal::is_specialization<test_return_type, std::optional>::value));
        EXPECT_TRUE((std::is_same_v<test_return_type::value_type, expected_result_type>));
    }
    {
        // ignore status -> return result only
        using wait_return_type = decltype(result.wait());
        EXPECT_TRUE((std::is_same_v<wait_return_type, expected_result_type>));
    }
    {
        // also return status -> pair<result, status>
        using wait_return_type = decltype(result.wait(status_out()));
        EXPECT_TRUE((std::is_same_v<wait_return_type::first_type, expected_result_type>));
        EXPECT_TRUE((std::is_same_v<wait_return_type::second_type, Status>));
    }
    {
        // also return status, but as out parameter -> result
        Status status;
        using wait_return_type = decltype(result.wait(status_out(status)));
        EXPECT_TRUE((std::is_same_v<wait_return_type, expected_result_type>));
    }
}

TEST_F(NonBlockingResultTest, owning_request_and_result_wait_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj = request();

    using RecvBufType = decltype(recv_buf_obj);

    auto buffers_on_heap = move_buffer_to_heap(std::move(recv_buf_obj));
    auto result          = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        std::move(buffers_on_heap)
    );
    EXPECT_EQ(num_wait_calls, 0);
    auto data = result.wait();
    EXPECT_EQ(num_wait_calls, 1);
    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data, expected_data);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, owning_request_and_result_wait_works_with_status_out) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj        = request();
    using RecvBufType       = decltype(recv_buf_obj);
    auto buffers_on_heap    = move_buffer_to_heap(std::move(recv_buf_obj));
    auto nonblocking_result = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        std::move(buffers_on_heap)
    );
    EXPECT_EQ(num_wait_calls, 0);
    auto [data, status] = nonblocking_result.wait(status_out());
    EXPECT_EQ(num_wait_calls, 1);
    EXPECT_EQ(status.tag(), TOUCHED_BY_MOCK_TAG);
    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data, expected_data);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, owning_request_and_result_wait_works_with_status_in) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj  = request();
    using RecvBufType = decltype(recv_buf_obj);
    auto result       = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj))
    );
    EXPECT_EQ(num_wait_calls, 0);
    Status status;
    auto   data = result.wait(status_out(status));
    EXPECT_EQ(num_wait_calls, 1);
    EXPECT_EQ(status.tag(), TOUCHED_BY_MOCK_TAG);
    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data, expected_data);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, owning_request_and_result_test_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj  = request();
    using RecvBufType = decltype(recv_buf_obj);
    auto result       = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj))
    );
    let_mpi_test_succeed = false;
    EXPECT_FALSE(result.test().has_value());
    let_mpi_test_succeed = true;
    auto data            = result.test();
    EXPECT_TRUE(data.has_value());
    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data.value(), expected_data);
}

TEST_F(NonBlockingResultTest, owning_request_and_result_test_works_status_out) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj  = request();
    using RecvBufType = decltype(recv_buf_obj);
    auto result       = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj))
    );
    let_mpi_test_succeed = false;
    EXPECT_FALSE(result.test(status_out()).has_value());
    let_mpi_test_succeed = true;
    auto data            = result.test(status_out());
    EXPECT_TRUE(data.has_value());
    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data.value().first, expected_data);
    EXPECT_EQ(data.value().second.tag(), TOUCHED_BY_MOCK_TAG);
}

TEST_F(NonBlockingResultTest, owning_request_and_result_test_works_status_in) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj  = request();
    using RecvBufType = decltype(recv_buf_obj);
    auto result       = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj))
    );
    let_mpi_test_succeed = false;
    Status status;
    EXPECT_FALSE(result.test(status_out(status)).has_value());
    let_mpi_test_succeed = true;
    auto data            = result.test(status_out(status));
    EXPECT_TRUE(data.has_value());
    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data.value(), expected_data);
    EXPECT_EQ(status.tag(), TOUCHED_BY_MOCK_TAG);
}

// TODO add nonowning recv buf
TEST_F(NonBlockingResultTest, owning_request_and_result_extract_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto request_obj        = request();
    using RecvBufType       = decltype(recv_buf_obj);
    auto nonblocking_result = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj))
    );

    auto [req, recv_buf] = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(req), Request>));
    EXPECT_TRUE((std::is_same_v<decltype(recv_buf), std::vector<int>>));

    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(recv_buf, expected_data);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_types_match) {
    auto request_obj = request();
    auto result      = kamping::internal::make_nonblocking_result(std::move(request_obj));

    EXPECT_TRUE(has_member_test_v<decltype(result)>);
    EXPECT_TRUE(has_member_wait_v<decltype(result)>);
    {
        // ignore status -> return bool, because we have no result
        using test_return_type = decltype(result.test());
        EXPECT_TRUE((std::is_same_v<test_return_type, bool>));
    }
    {
        // also return status -> optional<status>
        using test_return_type = decltype(result.test(status_out()));
        EXPECT_TRUE((internal::is_specialization<test_return_type, std::optional>::value));
        EXPECT_TRUE((std::is_same_v<test_return_type::value_type, Status>));
    }
    {
        // also return status, but as non-owning out parameter -> bool
        Status status;
        using test_return_type = decltype(result.test(status_out(status)));
        EXPECT_TRUE((std::is_same_v<test_return_type, bool>));
    }
    {
        // ignore status -> return nothing, because we have no result
        using wait_return_type = decltype(result.wait());
        EXPECT_TRUE((std::is_same_v<wait_return_type, void>));
    }
    {
        // also return status -> status
        using wait_return_type = decltype(result.wait(status_out()));
        EXPECT_TRUE((std::is_same_v<wait_return_type, Status>));
    }
    {
        // also return status, but as out parameter -> return nothing
        Status status;
        using wait_return_type = decltype(result.wait(status_out(status)));
        EXPECT_TRUE((std::is_same_v<wait_return_type, void>));
    }
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_test_works) {
    auto request_obj     = request();
    auto result          = kamping::internal::make_nonblocking_result(std::move(request_obj));
    let_mpi_test_succeed = false;
    EXPECT_FALSE(result.test());
    let_mpi_test_succeed = true;
    EXPECT_TRUE(result.test());
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_test_works_status_out) {
    auto request_obj     = request();
    auto result          = kamping::internal::make_nonblocking_result(std::move(request_obj));
    let_mpi_test_succeed = false;
    EXPECT_FALSE(result.test(status_out()));
    let_mpi_test_succeed         = true;
    std::optional<Status> status = result.test(status_out());
    EXPECT_TRUE(status.has_value());
    EXPECT_EQ(status.value().tag(), TOUCHED_BY_MOCK_TAG);
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_test_works_status_in) {
    auto request_obj     = request();
    auto result          = kamping::internal::make_nonblocking_result(std::move(request_obj));
    let_mpi_test_succeed = false;
    Status status;
    EXPECT_FALSE(result.test(status_out(status)));
    let_mpi_test_succeed = true;
    EXPECT_TRUE(result.test(status_out(status)));
    EXPECT_EQ(status.tag(), TOUCHED_BY_MOCK_TAG);
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_wait_works) {
    auto request_obj = request();
    auto result      = kamping::internal::make_nonblocking_result(std::move(request_obj));
    EXPECT_EQ(num_wait_calls, 0);
    static_assert(std::is_same_v<decltype(result.wait()), void>);
    result.wait();
    EXPECT_EQ(num_wait_calls, 1);
}

TEST_F(NonBlockingResultTest, owning_request_and_empty_result_extract_works) {
    auto request_obj        = request();
    auto nonblocking_result = kamping::internal::make_nonblocking_result(std::move(request_obj));
    auto req                = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(req), Request>));

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_types_match) {
    auto    recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    Request req;
    auto    request_obj = request(req);
    using RecvBufType   = decltype(recv_buf_obj);
    auto result         = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj))

    );
    EXPECT_FALSE(has_member_test_v<decltype(result)>)
        << "The result does not own the request, so test() should not be available.";
    EXPECT_FALSE(has_member_wait_v<decltype(result)>)
        << "The result does not own the request, so wait() should not be available.";
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_extract_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    Request req;
    auto    request_obj     = request(req);
    using RecvBufType       = decltype(recv_buf_obj);
    auto nonblocking_result = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj))
    );
    auto data = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(data), std::vector<int>>));

    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data, expected_data);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_with_buffer_other_than_recv_buf_extract_works) {
    auto recv_count_obj         = recv_count_out().construct_buffer_or_rebind();
    recv_count_obj.underlying() = 1;
    Request req;
    auto    request_obj     = request(req);
    using RecvCountType     = decltype(recv_count_obj);
    auto nonblocking_result = kamping::internal::make_nonblocking_result<std::tuple<RecvCountType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_count_obj))
    );
    auto [recv_count] = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(recv_count), int>));

    EXPECT_EQ(recv_count, 1);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_with_implicit_recv_buffer_extract_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    Request req;
    auto    request_obj     = request(req);
    using RequestType       = decltype(request_obj);
    auto nonblocking_result = kamping::internal::make_nonblocking_result<std::tuple<RequestType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj))

    );
    auto data = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(data), std::vector<int>>));

    auto expected_data = std::vector{42, 43, 44};
    EXPECT_EQ(data, expected_data);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_with_implicit_recv_buffer_and_recv_count_extract_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto recv_count_obj         = recv_count_out().construct_buffer_or_rebind();
    recv_count_obj.underlying() = 1;
    Request req;
    auto    request_obj     = request(req);
    using RecvBufType       = decltype(recv_buf_obj);
    using RecvCountType     = decltype(recv_count_obj);
    auto nonblocking_result = kamping::internal::make_nonblocking_result<std::tuple<RecvBufType, RecvCountType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj), std::move(recv_count_obj))
    );
    auto [recv_buf, recv_count] = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(recv_buf), std::vector<int>>));

    auto expected_recv_buf = std::vector{42, 43, 44};
    EXPECT_EQ(recv_buf, expected_recv_buf);
    EXPECT_EQ(recv_count, 1);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_with_recv_buffer_and_recv_count_extract_works) {
    auto recv_buf_obj = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    recv_buf_obj.underlying().push_back(42);
    recv_buf_obj.underlying().push_back(43);
    recv_buf_obj.underlying().push_back(44);
    auto recv_count_obj         = recv_count_out().construct_buffer_or_rebind();
    recv_count_obj.underlying() = 1;
    Request req;
    auto    request_obj     = request(req);
    using RecvBufType       = decltype(recv_buf_obj);
    using RecvCountType     = decltype(recv_count_obj);
    auto nonblocking_result = kamping::internal::make_nonblocking_result<std::tuple<RecvCountType, RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj), std::move(recv_count_obj))
    );
    auto [recv_count, recv_buf] = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(recv_buf), std::vector<int>>));
    EXPECT_TRUE((std::is_same_v<decltype(recv_count), int>));

    auto expected_recv_buf = std::vector{42, 43, 44};
    EXPECT_EQ(recv_buf, expected_recv_buf);
    EXPECT_EQ(recv_count, 1);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, non_owning_request_and_result_with_non_owning_recv_buffer_and_recv_count_extract_works) {
    std::vector<int> recv_buf_storage{42, 43, 44};
    auto             recv_buf_obj   = recv_buf(recv_buf_storage).construct_buffer_or_rebind();
    auto             recv_count_obj = recv_count_out().construct_buffer_or_rebind();
    recv_count_obj.underlying()     = 1;
    Request req;
    auto    request_obj     = request(req);
    using RecvBufType       = decltype(recv_buf_obj);
    using RecvCountType     = decltype(recv_count_obj);
    auto nonblocking_result = kamping::internal::make_nonblocking_result<std::tuple<RecvCountType, RecvBufType>>(
        std::move(request_obj),
        move_buffer_to_heap(std::move(recv_buf_obj), std::move(recv_count_obj))
    );
    auto [recv_count] = nonblocking_result.extract();
    EXPECT_TRUE((std::is_same_v<decltype(recv_count), int>));

    auto expected_recv_buf = std::vector{42, 43, 44};
    EXPECT_EQ(recv_buf_storage, expected_recv_buf);
    EXPECT_EQ(recv_count, 1);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(nonblocking_result.extract(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, wait_on_extracted_request) {
    auto request_obj = request();
    auto result      = kamping::internal::make_nonblocking_result(std::move(request_obj));
    auto req         = result.extract();
    (void)req;
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(result.wait(), "The result of this request has already been extracted.");
#endif
}

TEST_F(NonBlockingResultTest, test_on_extracted_request) {
    auto request_obj = request();
    auto result      = kamping::internal::make_nonblocking_result(std::move(request_obj));
    auto req         = result.extract();
    (void)req;
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(result.test(), "The result of this request has already been extracted.");
#endif
}
