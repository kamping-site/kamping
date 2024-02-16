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

#include <numeric>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/has_member.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"
#include "legacy_parameter_objects.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

namespace testing {
// Mock object with extract method
struct StructWithExtract {
    void extract() {}
};

// Mock object without extract method
struct StructWithoutExtract {};

// Test that receive buffers can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_buffer_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto recv_buffer = recv_buf(kamping::alloc_new<UnderlyingContainer>).get();
    static_assert(std::is_integral_v<typename decltype(recv_buffer)::value_type>, "Use integral Types in this test.");

    recv_buffer.resize(10);
    int* ptr = recv_buffer.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        std::move(recv_buffer),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_recv_buffer();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that receive counts can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_counts_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto recv_counts = recv_counts_out(alloc_new<UnderlyingContainer>).get();
    static_assert(std::is_integral_v<typename decltype(recv_counts)::value_type>, "Use integral Types in this test.");

    recv_counts.resize(10);
    int* ptr = recv_counts.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(recv_counts),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_recv_counts();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that the receive count can be moved into and extracted from a MPIResult object.
void test_recv_count_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;

    LibAllocatedSingleElementBuffer<int, ParameterType::recv_count, BufferType::out_buffer> recv_count_wrapper{};
    recv_count_wrapper.underlying() = 42;
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(recv_count_wrapper),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    int recv_count_value = mpi_result.extract_recv_count();
    EXPECT_EQ(recv_count_value, 42);
}

// Test that receive displs can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_displs_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto recv_displs = recv_displs_out(alloc_new<UnderlyingContainer>).get();
    static_assert(std::is_integral_v<typename decltype(recv_displs)::value_type>, "Use integral Types in this test.");

    recv_displs.resize(10);
    int* ptr = recv_displs.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(recv_displs),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_recv_displs();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that send counts can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_send_counts_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto send_counts = send_counts_out(alloc_new<UnderlyingContainer>).get();
    static_assert(std::is_integral_v<typename decltype(send_counts)::value_type>, "Use integral Types in this test.");

    send_counts.resize(10);
    int* ptr = send_counts.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(send_counts),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_send_counts();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that send count can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_send_count_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    LibAllocatedSingleElementBuffer<int, ParameterType::send_count, BufferType::out_buffer> send_count_wrapper{};
    send_count_wrapper.underlying() = 42;

    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(send_count_wrapper),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    int send_count = mpi_result.extract_send_count();
    EXPECT_EQ(send_count, 42);
}

// Test that send displs can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_send_displs_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto send_displs = send_displs_out(alloc_new<UnderlyingContainer>).get();
    static_assert(std::is_integral_v<typename decltype(send_displs)::value_type>, "Use integral Types in this test.");

    send_displs.resize(10);
    int* ptr = send_displs.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(send_displs),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_send_displs();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

} // namespace testing

TEST(MpiResultTest, has_extract_v_basics) {
    static_assert(
        has_extract_v<::testing::StructWithExtract>,
        "StructWithExtract contains extract() member function -> needs to be detected."
    );
    static_assert(
        !has_extract_v<::testing::StructWithoutExtract>,
        "StructWithoutExtract does not contain extract() member function."
    );
}

TEST(MpiResultTest, extract_recv_buffer_basics) {
    ::testing::test_recv_buffer_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_buffer_basics_own_container) {
    ::testing::test_recv_buffer_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_recv_counts_basics) {
    ::testing::test_recv_counts_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_counts_basics_own_container) {
    ::testing::test_recv_counts_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_recv_count_basics) {
    ::testing::test_recv_count_in_MPIResult();
}

TEST(MpiResultTest, extract_recv_displs_basics) {
    ::testing::test_recv_displs_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_displs_basics_own_container) {
    ::testing::test_recv_displs_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_send_counts_basics) {
    ::testing::test_send_counts_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_send_counts_basics_own_container) {
    ::testing::test_send_counts_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_send_displs_basics) {
    ::testing::test_send_displs_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_send_displs_basics_own_container) {
    ::testing::test_send_displs_in_MPIResult<::testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_send_recv_count) {
    using namespace kamping;
    using namespace kamping::internal;
    auto send_recv_count         = kamping::send_recv_count_out().get();
    send_recv_count.underlying() = 42;
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(send_recv_count),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    EXPECT_EQ(mpi_result.extract_send_recv_count(), 42);
}

TEST(MpiResultTest, extract_send_type) {
    using namespace kamping;
    using namespace kamping::internal;
    auto send_type         = kamping::send_type_out().get();
    send_type.underlying() = MPI_DOUBLE;
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(send_type),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    EXPECT_EQ(mpi_result.extract_send_type(), MPI_DOUBLE);
}

TEST(MpiResultTest, extract_recv_type) {
    using namespace kamping;
    using namespace kamping::internal;
    auto recv_type         = kamping::recv_type_out().get();
    recv_type.underlying() = MPI_CHAR;
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(recv_type),
        ResultCategoryNotUsed{}};
    EXPECT_EQ(mpi_result.extract_recv_type(), MPI_CHAR);
}

TEST(MpiResultTest, extract_send_recv_type) {
    using namespace kamping;
    using namespace kamping::internal;
    auto send_recv_type         = kamping::send_recv_type_out().get();
    send_recv_type.underlying() = MPI_CHAR;
    MPIResult mpi_result{
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        std::move(send_recv_type)};
    EXPECT_EQ(mpi_result.extract_send_recv_type(), MPI_CHAR);
}

TEST(MpiResultTest, extract_status_basics) {
    using namespace kamping;
    using namespace kamping::internal;
    auto status = status_out();

    status_param_to_native_ptr(status)->MPI_TAG = 42;
    MPIResult mpi_result{
        std::move(status),
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{},
        ResultCategoryNotUsed{}};
    auto underlying_status = mpi_result.extract_status();
    EXPECT_EQ(underlying_status.tag(), 42);
}

KAMPING_MAKE_HAS_MEMBER(extract_status)
KAMPING_MAKE_HAS_MEMBER(extract_recv_buffer)
KAMPING_MAKE_HAS_MEMBER(extract_recv_counts)
KAMPING_MAKE_HAS_MEMBER(extract_recv_count)
KAMPING_MAKE_HAS_MEMBER(extract_recv_displs)
KAMPING_MAKE_HAS_MEMBER(extract_send_counts)
KAMPING_MAKE_HAS_MEMBER(extract_send_count)
KAMPING_MAKE_HAS_MEMBER(extract_send_displs)
KAMPING_MAKE_HAS_MEMBER(extract_send_recv_count)
KAMPING_MAKE_HAS_MEMBER(extract_send_type)
KAMPING_MAKE_HAS_MEMBER(extract_recv_type)
KAMPING_MAKE_HAS_MEMBER(extract_send_recv_type)

TEST(MpiResultTest, removed_extract_functions) {
    using namespace ::kamping;
    using namespace ::kamping::internal;
    constexpr BufferType btype = BufferType::out_buffer;
    {
        // All of these should be extractable (used to make sure that the above macros work correctly)
        LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>                 status_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype> send_counts_sanity_check;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_sanity_check;
        LibAllocatedContainerBasedBuffer<int, ParameterType::recv_count, btype>               recv_count_sanity_check;
        LibAllocatedContainerBasedBuffer<int, ParameterType::send_count, btype>               send_count_sanity_check;
        LibAllocatedContainerBasedBuffer<int, ParameterType::send_recv_count, btype>    send_recv_count_sanity_check;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_type, btype> send_type_sanity_check;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::recv_type, btype> recv_type_sanity_check;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_recv_type, btype>
                           send_recv_type_sanity_check;
        kamping::MPIResult mpi_result_sanity_check{
            std::move(status_sanity_check),
            std::move(recv_buf_sanity_check),
            std::move(recv_counts_sanity_check),
            std::move(recv_count_sanity_check),
            std::move(recv_displs_sanity_check),
            std::move(send_counts_sanity_check),
            std::move(send_count_sanity_check),
            std::move(send_displs_sanity_check),
            std::move(send_recv_count_sanity_check),
            std::move(send_type_sanity_check),
            std::move(recv_type_sanity_check),
            std::move(send_recv_type_sanity_check)};
        EXPECT_TRUE(has_member_extract_status_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_count_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_count_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_recv_count_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_type_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_recv_type_v<decltype(mpi_result_sanity_check)>);
        EXPECT_TRUE(has_member_extract_send_recv_type_v<decltype(mpi_result_sanity_check)>);
        EXPECT_FALSE(decltype(mpi_result_sanity_check)::is_empty);
    }

    {
        // none of the extract function should work if the underlying buffer does not provide a member extract().
        kamping::MPIResult mpi_result{
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{},
            ResultCategoryNotUsed{}};
        EXPECT_FALSE(has_member_extract_status_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_counts_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_count_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_displs_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_counts_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_count_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_displs_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_recv_count_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_type_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_recv_type_v<decltype(mpi_result)>);
        EXPECT_FALSE(has_member_extract_send_recv_type_v<decltype(mpi_result)>);
        EXPECT_TRUE(decltype(mpi_result)::is_empty);
    }

    {
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_status;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_status;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_status;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype> send_counts_status;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_status;
        LibAllocatedContainerBasedBuffer<int, ParameterType::send_count, btype>               send_count;
        LibAllocatedContainerBasedBuffer<int, ParameterType::recv_count, btype>               recv_count;
        LibAllocatedContainerBasedBuffer<int, ParameterType::send_recv_count, btype>          send_recv_count;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_type, btype>       send_type;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::recv_type, btype>       recv_type;
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_recv_type, btype>  send_recv_type;
        auto result_status = make_mpi_result(
            std::move(recv_counts_status),
            std::move(recv_count),
            std::move(recv_displs_status),
            std::move(send_counts_status),
            std::move(send_count),
            std::move(send_displs_status),
            std::move(recv_buf_status),
            std::move(send_recv_count),
            std::move(send_type),
            std::move(recv_type),
            std::move(send_recv_type)
        );
        EXPECT_FALSE(has_member_extract_status_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_count_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_count_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_recv_count_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_type_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_recv_type_v<decltype(result_status)>);
        EXPECT_TRUE(has_member_extract_send_recv_type_v<decltype(result_status)>);
        EXPECT_FALSE(decltype(result_status)::is_empty);
    }

    {
        LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>                 status_recv_buf;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_recv_buf;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_recv_buf;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype> send_counts_recv_buf;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_recv_buf;
        auto result_recv_buf = make_mpi_result(
            std::move(status_recv_buf),
            std::move(recv_counts_recv_buf),
            std::move(recv_displs_recv_buf),
            std::move(send_displs_recv_buf),
            std::move(send_counts_recv_buf)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_recv_buf)>);
        EXPECT_FALSE(has_member_extract_recv_buffer_v<decltype(result_recv_buf)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_recv_buf)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_recv_buf)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_recv_buf)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_recv_buf)>);
        EXPECT_FALSE(decltype(result_recv_buf)::is_empty);
    }

    {
        LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>                 status_recv_counts;
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_recv_counts;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_recv_counts;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype> send_counts_recv_counts;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_recv_counts;
        auto result_recv_counts = make_mpi_result(
            std::move(status_recv_counts),
            std::move(recv_buf_recv_counts),
            std::move(recv_displs_recv_counts),
            std::move(send_counts_recv_counts),
            std::move(send_displs_recv_counts)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_recv_counts)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_recv_counts)>);
        EXPECT_FALSE(has_member_extract_recv_counts_v<decltype(result_recv_counts)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_recv_counts)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_recv_counts)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_recv_counts)>);
        EXPECT_FALSE(decltype(result_recv_counts)::is_empty);
    }

    {
        LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>                 status_recv_displs;
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_recv_displs;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_recv_displs;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype> send_counts_recv_displs;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_recv_displs;
        auto result_recv_displs = make_mpi_result(
            std::move(status_recv_displs),
            std::move(recv_buf_recv_displs),
            std::move(recv_counts_recv_displs),
            std::move(send_counts_recv_displs),
            std::move(send_displs_recv_displs)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_recv_displs)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_recv_displs)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_recv_displs)>);
        EXPECT_FALSE(has_member_extract_recv_displs_v<decltype(result_recv_displs)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_recv_displs)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_recv_displs)>);
        EXPECT_FALSE(decltype(result_recv_displs)::is_empty);
    }

    {
        LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>                 status_send_counts;
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_send_counts;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_send_counts;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_send_counts;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, btype> send_displs_send_counts;
        auto result_send_counts = make_mpi_result(
            std::move(status_send_counts),
            std::move(recv_buf_send_counts),
            std::move(recv_counts_send_counts),
            std::move(recv_displs_send_counts),
            std::move(send_displs_send_counts)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_send_counts)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_send_counts)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_send_counts)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_send_counts)>);
        EXPECT_FALSE(has_member_extract_send_counts_v<decltype(result_send_counts)>);
        EXPECT_TRUE(has_member_extract_send_displs_v<decltype(result_send_counts)>);
        EXPECT_FALSE(decltype(result_send_counts)::is_empty);
    }

    {
        LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>                 status_send_displs;
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf_send_displs;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts_send_displs;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs_send_displs;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, btype> send_counts_send_displs;
        auto result_send_displs = make_mpi_result(
            std::move(status_send_displs),
            std::move(recv_buf_send_displs),
            std::move(recv_counts_send_displs),
            std::move(recv_displs_send_displs),
            std::move(send_counts_send_displs)
        );
        EXPECT_TRUE(has_member_extract_status_v<decltype(result_send_displs)>);
        EXPECT_TRUE(has_member_extract_recv_buffer_v<decltype(result_send_displs)>);
        EXPECT_TRUE(has_member_extract_recv_counts_v<decltype(result_send_displs)>);
        EXPECT_TRUE(has_member_extract_recv_displs_v<decltype(result_send_displs)>);
        EXPECT_TRUE(has_member_extract_send_counts_v<decltype(result_send_displs)>);
        EXPECT_FALSE(has_member_extract_send_displs_v<decltype(result_send_displs)>);
        EXPECT_FALSE(decltype(result_send_displs)::is_empty);
    }
}

TEST(MakeMpiResultTest, pass_random_order_buffer) {
    {
        constexpr BufferType btype = BufferType::out_buffer;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts;
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs;
        LibAllocatedSingleElementBuffer<Status, ParameterType::status, btype>                 status;
        status_param_to_native_ptr(status)->MPI_TAG = 42;

        auto result =
            make_mpi_result(std::move(recv_counts), std::move(status), std::move(recv_buf), std::move(recv_displs));

        auto result_recv_buf    = result.extract_recv_buffer();
        auto result_recv_counts = result.extract_recv_counts();
        auto result_recv_displs = result.extract_recv_displs();
        auto result_status      = result.extract_status();

        static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, char>);
        static_assert(std::is_same_v<decltype(result_recv_counts)::value_type, int>);
        static_assert(std::is_same_v<decltype(result_recv_displs)::value_type, int>);
        ASSERT_EQ(result_status.tag(), 42);
    }
    {
        constexpr BufferType btype = BufferType::out_buffer;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts;
        LibAllocatedContainerBasedBuffer<std::vector<double>, ParameterType::recv_buf, btype> recv_buf;

        auto result = make_mpi_result(std::move(recv_counts), std::move(recv_buf));

        auto result_recv_buf    = result.extract_recv_buffer();
        auto result_recv_counts = result.extract_recv_counts();

        static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, double>);
        static_assert(std::is_same_v<decltype(result_recv_counts)::value_type, int>);
    }
}

TEST(MakeMpiResultTest, pass_send_recv_buf) {
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_recv_buf, BufferType::in_out_buffer>
         send_recv_buf;
    auto result          = make_mpi_result(std::move(send_recv_buf));
    auto result_recv_buf = result.extract_recv_buffer();
    static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, int>);
}

TEST(MakeMpiResultTest, check_content) {
    constexpr BufferType btype = BufferType::out_buffer;

    std::vector<int> recv_buf_data(20);
    std::iota(recv_buf_data.begin(), recv_buf_data.end(), 0);
    Span<int> recv_buf_container = {recv_buf_data.data(), recv_buf_data.size()};
    LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_buf, btype> recv_buf(recv_buf_container);

    std::vector<int> recv_counts_data(20);
    std::iota(recv_counts_data.begin(), recv_counts_data.end(), 20);
    Span<int> recv_counts_container = {recv_counts_data.data(), recv_counts_data.size()};
    LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_counts, btype> recv_counts(recv_counts_container);

    std::vector<int> recv_displs_data(20);
    std::iota(recv_displs_data.begin(), recv_displs_data.end(), 40);
    Span<int> recv_displs_container = {recv_displs_data.data(), recv_displs_data.size()};
    LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::recv_displs, btype> recv_displs(recv_displs_container);

    std::vector<int> send_displs_data(20);
    std::iota(send_displs_data.begin(), send_displs_data.end(), 60);
    Span<int> send_displs_container = {send_displs_data.data(), send_displs_data.size()};
    LibAllocatedContainerBasedBuffer<Span<int>, ParameterType::send_displs, btype> send_displs(send_displs_container);

    auto result =
        make_mpi_result(std::move(recv_buf), std::move(recv_counts), std::move(recv_displs), std::move(send_displs));

    auto result_recv_buf = result.extract_recv_buffer();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_buf.data()[i], i);
    }
    auto result_recv_counts = result.extract_recv_counts();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_counts.data()[i], i + 20);
    }
    auto result_recv_displs = result.extract_recv_displs();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_recv_displs.data()[i], i + 40);
    }
    auto result_send_displs = result.extract_send_displs();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_EQ(result_send_displs.data()[i], i + 60);
    }
}
