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

#include <type_traits>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/parameter_objects.hpp"

using namespace ::kamping::internal;

// Test our minimal span implementation
TEST(Test_Span, basic_functionality) {
    std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    Span<int> int_span(values.data(), values.size());
    EXPECT_EQ(values.size(), int_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), int_span.size_bytes());
    EXPECT_FALSE(int_span.empty());
    EXPECT_EQ(values.data(), int_span.data());
    EXPECT_EQ(int_span.data(), int_span.data());

    Span<int> tuple_constructed_span(std::tuple<int*, size_t>{values.data(), values.size()});
    EXPECT_EQ(values.size(), tuple_constructed_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), tuple_constructed_span.size_bytes());
    EXPECT_FALSE(tuple_constructed_span.empty());
    EXPECT_EQ(values.data(), tuple_constructed_span.data());
    EXPECT_EQ(tuple_constructed_span.data(), int_span.data());

    Span<const int> const_int_span = {values.data(), values.size()};
    EXPECT_EQ(values.size(), const_int_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), const_int_span.size_bytes());
    EXPECT_FALSE(const_int_span.empty());
    EXPECT_EQ(values.data(), const_int_span.data());
    EXPECT_EQ(const_int_span.data(), int_span.data());

    Span<int> empty_span = {values.data(), 0};
    EXPECT_TRUE(empty_span.empty());
    EXPECT_EQ(0, empty_span.size());
    EXPECT_EQ(0, empty_span.size_bytes());
    EXPECT_EQ(values.data(), empty_span.data());

    Span<int> nullptr_span = {nullptr, 0};
    EXPECT_TRUE(nullptr_span.empty());
    EXPECT_EQ(0, nullptr_span.size());
    EXPECT_EQ(0, nullptr_span.size_bytes());
    EXPECT_EQ(nullptr, nullptr_span.data());

    static_assert(
        std::is_pointer_v<decltype(int_span.data())>,
        "Member data() of internal::Span<T*, size_t> does not return a pointer.");
    static_assert(
        std::is_pointer_v<decltype(const_int_span.data())>,
        "Member data() of internal::Span<T const *, size_t> does not return a pointer.");
    static_assert(
        std::is_const_v<std::remove_pointer_t<decltype(const_int_span.data())>>,
        "Member data() of internal::Span<T const *, size_t> does not return a pointer pointing to const memory.");
    static_assert(
        !std::is_const_v<std::remove_pointer_t<decltype(int_span.data())>>,
        "Member data() of internal::Span<T*, size_t> does return a pointer pointing to const memory.");

    static_assert(
        std::is_same_v<decltype(int_span)::value_type, decltype(values)::value_type>,
        "Member value_type of internal::Span<T*, size_t> does not match the element type of the underlying container.");
    static_assert(
        std::is_same_v<decltype(int_span)::size_type, decltype(values)::size_type>,
        "Member size_type of internal::Span<T*, size_t> does not match the element type of the underlying container.");
    static_assert(
        std::is_same_v<decltype(int_span)::difference_type, decltype(values)::difference_type>,
        "Member difference_type of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container.");
    static_assert(
        std::is_same_v<decltype(int_span)::pointer, decltype(values)::pointer>,
        "Member pointer of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container.");
    static_assert(
        std::is_same_v<decltype(int_span)::const_pointer, decltype(values)::const_pointer>,
        "Member const_pointer of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container.");
    static_assert(
        std::is_same_v<decltype(int_span)::reference, decltype(values)::reference>,
        "Member reference of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container.");
    static_assert(
        std::is_same_v<decltype(int_span)::const_reference, decltype(values)::const_reference>,
        "Member const_reference of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container.");
}

// Tests the basic functionality of ContainerBasedConstBuffer (i.e. its only public function get())
TEST(ContainerBasedConstBufferTest, get_basics) {
    std::vector<int>       int_vec{1, 2, 3};
    std::vector<int> const int_vec_const{1, 2, 3, 4};

    constexpr ParameterType                            ptype = ParameterType::send_counts;
    ContainerBasedConstBuffer<std::vector<int>, ptype> buffer_based_on_int_vector(int_vec);
    ContainerBasedConstBuffer<std::vector<int>, ptype> buffer_based_on_const_int_vector(int_vec_const);

    EXPECT_EQ(buffer_based_on_int_vector.get().size(), int_vec.size());
    EXPECT_EQ(buffer_based_on_int_vector.get().data(), int_vec.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_int_vector.get().data()), const int*>);

    EXPECT_EQ(buffer_based_on_const_int_vector.get().size(), int_vec_const.size());
    EXPECT_EQ(buffer_based_on_const_int_vector.get().data(), int_vec_const.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_const_int_vector.get().data()), const int*>);
}

TEST(ContainerBasedConstBufferTest, get_containers_other_than_vector) {
    std::string                                                  str = "I am underlying storage";
    testing::OwnContainer<int>                                   own_container;
    constexpr ParameterType                                      ptype = ParameterType::send_counts;
    ContainerBasedConstBuffer<std::string, ptype>                buffer_based_on_string(str);
    ContainerBasedConstBuffer<testing::OwnContainer<int>, ptype> buffer_based_on_own_container(own_container);

    EXPECT_EQ(buffer_based_on_string.get().size(), str.size());
    EXPECT_EQ(buffer_based_on_string.get().data(), str.data());

    EXPECT_EQ(buffer_based_on_own_container.get().size(), own_container.size());
    EXPECT_EQ(buffer_based_on_own_container.get().data(), own_container.data());
}

TEST(UserAllocatedContainerBasedBufferTest, get_ptr_basics) {
    std::vector<int> int_vec{1, 2, 3, 2, 1};

    constexpr ParameterType                                    ptype = ParameterType::send_counts;
    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype> buffer_based_on_int_vector(int_vec);

    auto resize_write_check = [&](size_t requested_size) {
        buffer_based_on_int_vector.resize(requested_size);
        int* ptr = buffer_based_on_int_vector.data();
        EXPECT_EQ(ptr, int_vec.data());
        EXPECT_EQ(int_vec.size(), requested_size);
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
            EXPECT_EQ(ptr[i], int_vec[i]);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    resize_write_check(9);
}

TEST(UserAllocatedContainerBasedBufferTest, get_ptr_containers_other_than_vector) {
    testing::OwnContainer<int> own_container;

    constexpr ParameterType                                              ptype = ParameterType::recv_counts;
    UserAllocatedContainerBasedBuffer<testing::OwnContainer<int>, ptype> buffer_based_on_own_container(own_container);

    auto resize_write_check = [&](size_t requested_size) {
        buffer_based_on_own_container.resize(requested_size);
        int* ptr = buffer_based_on_own_container.data();
        EXPECT_EQ(ptr, own_container.data());
        EXPECT_EQ(own_container.size(), requested_size);
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
            EXPECT_EQ(ptr[i], own_container[i]);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    resize_write_check(9);
}

TEST(LibAllocatedContainerBasedBufferTest, get_ptr_extract_basics) {
    constexpr ParameterType                                   ptype = ParameterType::recv_counts;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ptype> buffer_based_on_int_vector;

    auto resize_write_check = [&](size_t requested_size) {
        buffer_based_on_int_vector.resize(requested_size);
        int* ptr = buffer_based_on_int_vector.data();
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    const size_t last_resize = 9;
    resize_write_check(last_resize);
    std::vector<int> underlying_container = buffer_based_on_int_vector.extract();
    for (size_t i = 0; i < last_resize; ++i) {
        EXPECT_EQ(underlying_container[i], static_cast<int>(last_resize - i));
    }
}

TEST(LibAllocatedContainerBasedBufferTest, get_ptr_extract_containers_other_than_vector) {
    constexpr ParameterType                                             ptype = ParameterType::recv_counts;
    LibAllocatedContainerBasedBuffer<testing::OwnContainer<int>, ptype> buffer_based_on_own_container;

    auto resize_write_check = [&](size_t requested_size) {
        buffer_based_on_own_container.resize(requested_size);
        int* ptr = buffer_based_on_own_container.data();
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    const size_t last_resize = 9;
    resize_write_check(last_resize);
    testing::OwnContainer<int> underlying_container = buffer_based_on_own_container.extract();
    for (size_t i = 0; i < last_resize; ++i) {
        EXPECT_EQ(underlying_container[i], static_cast<int>(last_resize - i));
    }
}

TEST(SingleElementConstBufferTest, get_basics) {
    constexpr ParameterType              ptype = ParameterType::send_counts;
    int                                  value = 5;
    SingleElementConstBuffer<int, ptype> int_buffer(value);

    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_FALSE(int_buffer.is_modifiable);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, decltype(value)>);
}
