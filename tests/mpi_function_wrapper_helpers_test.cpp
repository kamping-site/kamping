// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
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
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
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
    auto recv_buffer = recv_buf(kamping::NewContainer<UnderlyingContainer>{});
    static_assert(std::is_integral_v<typename decltype(recv_buffer)::value_type>, "Use integral Types in this test.");

    recv_buffer.resize(10);
    int* ptr = recv_buffer.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        std::move(recv_buffer),
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{}};
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
    auto recv_counts = recv_counts_out(NewContainer<UnderlyingContainer>{});
    static_assert(std::is_integral_v<typename decltype(recv_counts)::value_type>, "Use integral Types in this test.");

    recv_counts.resize(10);
    int* ptr = recv_counts.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        BufferCategoryNotUsed{},
        std::move(recv_counts),
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_recv_counts();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that the receive count can be moved into and extracted from a MPIResult object.
void test_recv_count_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;

    LibAllocatedSingleElementBuffer<int, ParameterType::recv_counts, BufferType::in_buffer> recv_count_wrapper{};
    *recv_count_wrapper.get().data() = 42;
    MPIResult mpi_result{
        BufferCategoryNotUsed{},
        std::move(recv_count_wrapper),
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{}};
    int recv_count_value = mpi_result.extract_recv_counts();
    EXPECT_EQ(recv_count_value, 42);
}

// Test that receive displs can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_recv_displs_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto recv_displs = recv_displs_out(NewContainer<UnderlyingContainer>{});
    static_assert(std::is_integral_v<typename decltype(recv_displs)::value_type>, "Use integral Types in this test.");

    recv_displs.resize(10);
    int* ptr = recv_displs.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{},
        std::move(recv_displs),
        BufferCategoryNotUsed{}};
    UnderlyingContainer underlying_container = mpi_result.extract_recv_displs();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}

// Test that send displs can be moved into and extracted from a MPIResult object.
template <typename UnderlyingContainer>
void test_send_displs_in_MPIResult() {
    using namespace kamping;
    using namespace kamping::internal;
    auto send_displs = send_displs_out(NewContainer<UnderlyingContainer>{});
    static_assert(std::is_integral_v<typename decltype(send_displs)::value_type>, "Use integral Types in this test.");

    send_displs.resize(10);
    int* ptr = send_displs.data();
    std::iota(ptr, ptr + 10, 0);
    MPIResult mpi_result{
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{},
        BufferCategoryNotUsed{},
        std::move(send_displs)};
    UnderlyingContainer underlying_container = mpi_result.extract_send_displs();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(underlying_container[i], i);
    }
}
} // namespace testing

TEST(MpiResultTest, has_extract_v_basics) {
    static_assert(
        has_extract_v<testing::StructWithExtract>,
        "StructWithExtract contains extract() member function -> needs to be detected."
    );
    static_assert(
        !has_extract_v<testing::StructWithoutExtract>,
        "StructWithoutExtract does not contain extract() member function."
    );
}

TEST(MpiResultTest, extract_recv_buffer_basics) {
    testing::test_recv_buffer_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_buffer_basics_own_container) {
    testing::test_recv_buffer_in_MPIResult<testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_recv_counts_basics) {
    testing::test_recv_counts_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_counts_basics_own_container) {
    testing::test_recv_counts_in_MPIResult<testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_recv_count_basics) {
    testing::test_recv_count_in_MPIResult();
}

TEST(MpiResultTest, extract_recv_displs_basics) {
    testing::test_recv_displs_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_recv_displs_basics_own_container) {
    testing::test_recv_displs_in_MPIResult<testing::OwnContainer<int>>();
}

TEST(MpiResultTest, extract_send_displs_basics) {
    testing::test_send_displs_in_MPIResult<std::vector<int>>();
}

TEST(MpiResultTest, extract_send_displs_basics_own_container) {
    testing::test_send_displs_in_MPIResult<testing::OwnContainer<int>>();
}

TEST(MakeMpiResultTest, pass_random_order_buffer) {
    {
        constexpr BufferType btype = BufferType::in_buffer;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts;
        LibAllocatedContainerBasedBuffer<std::vector<char>, ParameterType::recv_buf, btype>   recv_buf;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, btype> recv_displs;

        auto result = make_mpi_result(std::move(recv_counts), std::move(recv_buf), std::move(recv_displs));

        auto result_recv_buf    = result.extract_recv_buffer();
        auto result_recv_counts = result.extract_recv_counts();
        auto result_recv_displs = result.extract_recv_displs();

        static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, char>);
        static_assert(std::is_same_v<decltype(result_recv_counts)::value_type, int>);
        static_assert(std::is_same_v<decltype(result_recv_displs)::value_type, int>);
    }
    {
        constexpr BufferType btype = BufferType::in_buffer;
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, btype> recv_counts;
        LibAllocatedContainerBasedBuffer<std::vector<double>, ParameterType::recv_buf, btype> recv_buf;

        auto result = make_mpi_result(std::move(recv_counts), std::move(recv_buf));

        auto result_recv_buf    = result.extract_recv_buffer();
        auto result_recv_counts = result.extract_recv_counts();

        static_assert(std::is_same_v<decltype(result_recv_buf)::value_type, double>);
        static_assert(std::is_same_v<decltype(result_recv_counts)::value_type, int>);
    }
}

TEST(MakeMpiResultTest, check_content) {
    constexpr BufferType btype = BufferType::in_buffer;

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
