// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
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

#include "kamping/buffer_factories.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

namespace testing {
template <typename ExpectedValueType, typename GeneratedBuffer, typename T>
void test_const_buffer(
    const GeneratedBuffer& generated_buffer, kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::Span<T>& expected_span) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_FALSE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::ptype, expected_parameter_type);

    auto span = generated_buffer.get();
    static_assert(std::is_pointer_v<decltype(span.ptr)>, "Member ptr of internal::Span is not a pointer.");
    static_assert(
        std::is_const_v<std::remove_pointer_t<decltype(span.ptr)>>,
        "Member ptr of internal::Span does not point to const memory.");

    EXPECT_EQ(span.ptr, expected_span.ptr);
    EXPECT_EQ(span.size, expected_span.size);
    // TODO redundant?
    for (size_t i = 0; i < expected_span.size; ++i) {
        EXPECT_EQ(span.ptr[i], expected_span.ptr[i]);
    }
}

template <typename ExpectedValueType, typename GeneratedBuffer, typename UnderlyingContainer>
void test_user_allocated_buffer(
    GeneratedBuffer& generated_buffer, kamping::internal::ParameterType expected_parameter_type,
    UnderlyingContainer& underlying_container) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_TRUE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::ptype, expected_parameter_type);

    auto resize_write_check = [&](size_t nb_elements) {
        ExpectedValueType* ptr = generated_buffer.get_ptr(nb_elements);
        EXPECT_EQ(ptr, std::data(underlying_container));
        for (size_t i = 0; i < nb_elements; ++i) {
            ptr[i] = static_cast<ExpectedValueType>(nb_elements - i);
            EXPECT_EQ(ptr[i], underlying_container[i]);
        }
    };
    resize_write_check(10);
    resize_write_check(30);
    resize_write_check(5);
}

template <typename ExpectedValueType, typename GeneratedBuffer>
void test_library_allocated_buffer(
    GeneratedBuffer& generated_buffer, kamping::internal::ParameterType expected_parameter_type) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_TRUE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::ptype, expected_parameter_type);

    // TODO how to test this?
    std::ignore = generated_buffer.get_ptr(10);
    std::ignore = generated_buffer.get_ptr(30);
    std::ignore = generated_buffer.get_ptr(5);
}

} // namespace testing

TEST(HelpersTest, send_buf_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_buf(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_int_vec, ParameterType::send_buf, expected_span);
}

TEST(HelpersTest, send_buf_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_buf(const_int_vec);
    Span<int>              expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_const_int_vec, ParameterType::send_buf, expected_span);
}

TEST(HelpersTest, send_counts_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_counts(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_int_vec, ParameterType::send_counts, expected_span);
}

TEST(HelpersTest, send_counts_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_counts(const_int_vec);
    Span<int>              expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_const_int_vec, ParameterType::send_counts, expected_span);
}

TEST(HelpersTest, recv_counts_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = recv_counts_in(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_int_vec, ParameterType::recv_counts, expected_span);
}

TEST(HelpersTest, recv_counts_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = recv_counts_in(const_int_vec);
    Span<int>              expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_const_int_vec, ParameterType::recv_counts, expected_span);
}

TEST(HelpersTest, send_displs_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_displs_in(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_int_vec, ParameterType::send_displs, expected_span);
}

TEST(HelpersTest, send_displs_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_displs_in(const_int_vec);
    Span<int>              expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_const_int_vec, ParameterType::send_displs, expected_span);
}

TEST(HelpersTest, recv_displs_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = recv_displs_in(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_int_vec, ParameterType::recv_displs, expected_span);
}

TEST(HelpersTest, recv_displs_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = recv_displs_in(const_int_vec);
    Span<int>              expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(gen_via_const_int_vec, ParameterType::recv_displs, expected_span);
}

TEST(HelpersTest, recv_buf_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_on_user_alloc_vector = recv_buf(int_vec);
    using ExpectedValueType                      = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector, ParameterType::recv_buf, int_vec);
}

TEST(HelpersTest, recv_buf_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = recv_buf(NewContainer<std::vector<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector, ParameterType::recv_buf);
}

TEST(HelpersTest, send_displs_out_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_vector = send_displs_out(int_vec);
    using ExpectedValueType                            = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector, ParameterType::send_displs, int_vec);
}

TEST(HelpersTest, send_displs_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = send_displs_out(NewContainer<std::vector<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector, ParameterType::send_displs);
}

TEST(HelpersTest, recv_counts_out_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_buffer    = recv_counts_out(int_vec);
    auto             buffer_based_on_library_alloc_vector = recv_counts_out(NewContainer<std::vector<int>>{});
    using ExpectedValueType                               = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_buffer, ParameterType::recv_counts, int_vec);
}

TEST(HelpersTest, recv_counts_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = recv_counts_out(NewContainer<std::vector<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector, ParameterType::recv_counts);
}

TEST(HelpersTest, recv_displs_out_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_vector = recv_displs_out(int_vec);
    using ExpectedValueType                            = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector, ParameterType::recv_displs, int_vec);
}

TEST(HelpersTest, recv_displs_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = recv_displs_out(NewContainer<std::vector<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector, ParameterType::recv_displs);
}
