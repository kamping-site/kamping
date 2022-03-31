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

#include "helpers_for_testing.hpp"
#include <type_traits>

#include <gtest/gtest.h>

#include "kamping/parameter_objects.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

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

TEST(UserAllocatedContainerBasedBufferTest, resize_and_data_basics) {
    std::vector<int> int_vec{1, 2, 3, 2, 1};

    constexpr ParameterType                                    ptype = ParameterType::send_counts;
    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype> buffer_based_on_int_vector(int_vec);
    EXPECT_EQ(int_vec.size(), buffer_based_on_int_vector.get().size());
    EXPECT_EQ(int_vec.data(), buffer_based_on_int_vector.get().data());

    auto resize_write_check = [&](size_t requested_size) {
        buffer_based_on_int_vector.resize(requested_size);
        int* ptr = buffer_based_on_int_vector.data();
        EXPECT_EQ(ptr, int_vec.data());
        EXPECT_EQ(int_vec.data(), buffer_based_on_int_vector.get().data());
        EXPECT_EQ(int_vec.size(), requested_size);
        EXPECT_EQ(int_vec.size(), buffer_based_on_int_vector.get().size());
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
            EXPECT_EQ(ptr[i], int_vec[i]);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    resize_write_check(9);
}

TEST(UserAllocatedContainerBasedBufferTest, resize_and_data_containers_other_than_vector) {
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

TEST(LibAllocatedContainerBasedBufferTest, resize_and_data_extract_basics) {
    constexpr ParameterType                                   ptype = ParameterType::recv_counts;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ptype> buffer_based_on_int_vector;

    auto resize_write_check = [&](size_t requested_size) {
        buffer_based_on_int_vector.resize(requested_size);
        EXPECT_EQ(buffer_based_on_int_vector.size(), requested_size);
        EXPECT_EQ(buffer_based_on_int_vector.get().size(), requested_size);
        int* ptr = buffer_based_on_int_vector.data();
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    const size_t last_resize = 9;
    resize_write_check(last_resize);

    // The buffer will be in an invalid state after extraction; that's why we have to access these attributes
    // beforehand.
    const auto       size_of_buffer        = buffer_based_on_int_vector.size();
    const auto       data_of_buffer        = buffer_based_on_int_vector.data();
    const auto       size_of_get_of_buffer = buffer_based_on_int_vector.get().size();
    const auto       data_of_get_of_buffer = buffer_based_on_int_vector.get().data();
    std::vector<int> underlying_container  = buffer_based_on_int_vector.extract();
    EXPECT_EQ(underlying_container.size(), size_of_buffer);
    EXPECT_EQ(underlying_container.size(), size_of_get_of_buffer);
    EXPECT_EQ(underlying_container.data(), data_of_buffer);
    EXPECT_EQ(underlying_container.data(), data_of_get_of_buffer);
    for (size_t i = 0; i < last_resize; ++i) {
        EXPECT_EQ(underlying_container[i], static_cast<int>(last_resize - i));
    }
}

TEST(LibAllocatedContainerBasedBufferTest, extract_containers_other_than_vector) {
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

TEST(UserAllocatedContainerBasedBufferTest, resize_user_allocated_buffer) {
    std::vector<int>        data(20, 0);
    Span<int>               container = {data.data(), data.size()};
    constexpr ParameterType ptype     = ParameterType::send_counts;

    UserAllocatedContainerBasedBuffer<Span<int>, ptype> span_buffer(container);

    for (size_t i = 0; i <= 20; ++i) {
        span_buffer.resize(i);
        EXPECT_EQ(20, span_buffer.size());
    }

    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype> vec_buffer(data);

    for (size_t i = 0; i <= 20; ++i) {
        vec_buffer.resize(i);
        EXPECT_EQ(i, vec_buffer.size());
    }
}
