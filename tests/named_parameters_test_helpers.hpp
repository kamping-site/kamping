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

#include <type_traits>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "helpers_for_testing.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"
#include "legacy_parameter_objects.hpp"

namespace testing {
template <typename ExpectedValueType, typename GeneratedBuffer, typename T>
void test_const_buffer(
    GeneratedBuffer const&           generated_buffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    kamping::Span<T>&                expected_span
) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_FALSE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    auto span = generated_buffer.get();
    static_assert(std::is_pointer_v<decltype(span.data())>, "Member ptr of internal::Span is not a pointer.");
    static_assert(
        std::is_const_v<std::remove_pointer_t<decltype(span.data())>>,
        "Member data() of internal::Span does not point to const memory."
    );

    EXPECT_EQ(span.data(), expected_span.data());
    EXPECT_EQ(span.size(), expected_span.size());
    // TODO redundant?
    for (size_t i = 0; i < expected_span.size(); ++i) {
        EXPECT_EQ(span.data()[i], expected_span.data()[i]);
    }
}

template <typename ExpectedValueType, typename GeneratedBuffer, typename ExpectedValueContainer>
void test_owning_buffer(
    GeneratedBuffer const&           generated_buffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    ExpectedValueContainer&&         expected_value_container
) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_FALSE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    auto span = generated_buffer.get();
    static_assert(std::is_pointer_v<decltype(span.data())>, "Member ptr of internal::Span is not a pointer.");
    static_assert(
        std::is_const_v<std::remove_pointer_t<decltype(span.data())>>,
        "Member data() of internal::Span does not point to const memory."
    );

    EXPECT_EQ(span.size(), expected_value_container.size());
    for (size_t i = 0; i < expected_value_container.size(); ++i) {
        EXPECT_EQ(span.data()[i], expected_value_container[i]);
    }
}

template <typename ExpectedValueType, typename GeneratedBuffer, typename T>
void test_modifiable_buffer(
    GeneratedBuffer&                 generated_buffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    kamping::Span<T>&                expected_span
) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_TRUE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    auto span = generated_buffer.get();
    static_assert(std::is_pointer_v<decltype(span.data())>, "Member ptr of internal::Span is not a pointer.");
    static_assert(
        !std::is_const_v<std::remove_pointer_t<decltype(span.data())>>,
        "Member data() of internal::Span does point to const memory."
    );

    EXPECT_EQ(span.data(), expected_span.data());
    EXPECT_EQ(span.size(), expected_span.size());
    // TODO redundant?
    for (size_t i = 0; i < expected_span.size(); ++i) {
        EXPECT_EQ(span.data()[i], expected_span.data()[i]);
    }
}

template <typename ExpectedValueType, typename GeneratedBuffer, typename UnderlyingContainer>
void test_user_allocated_buffer(
    GeneratedBuffer&                 generated_buffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    UnderlyingContainer&             underlying_container
) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_TRUE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    auto resize_write_check = [&](size_t nb_elements) {
        generated_buffer.resize(nb_elements);
        ExpectedValueType* ptr = generated_buffer.data();
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
    GeneratedBuffer&                 generated_buffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type
) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_TRUE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    // TODO How can we test if the underlying storage resizes correctly to x elements when calling
    // generated_buffer.resize(x)?
    for (size_t size: std::vector<size_t>{10, 30, 5}) {
        generated_buffer.resize(size);
        EXPECT_EQ(generated_buffer.size(), size);
    }
}

template <typename ExpectedValueType, typename GeneratedBuffer>
void test_single_element_buffer(
    GeneratedBuffer const&           generatedbuffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    ExpectedValueType const          value,
    bool                             should_be_modifiable = false
) {
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_EQ(GeneratedBuffer::is_modifiable, should_be_modifiable);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    auto get_result = generatedbuffer.get();
    EXPECT_EQ(get_result.size(), 1);
    EXPECT_EQ(*(get_result.data()), value);
}
} // namespace testing
