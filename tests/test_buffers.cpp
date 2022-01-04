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

#include "kamping/buffers.hpp"

using namespace ::kamping::internal;

// Simple container to test KaMPI.ng's buffers with containers other than std::vector
template <typename T>
class OwnContainer {
public:
    using value_type = T;
    T* data() noexcept {
        return _vec.data();
    }
    const T* data() const noexcept {
        return _vec.data();
    }
    std::size_t size() const {
        return _vec.size();
    }
    void resize(std::size_t new_size) {
        _vec.resize(new_size);
    }
    const T& operator[](size_t i) const {
        return _vec[i];
    }
    T& operator[](size_t i) {
        return _vec[i];
    }

private:
    std::vector<T> _vec;
};

// Tests the basic functionality of ContainerBasedConstBuffer (i.e. its only public function get())
TEST(Test_ContainerBasedConstBuffer, get_basics) {
    std::vector<int>       int_vec{1, 2, 3};
    std::vector<int> const int_vec_const{1, 2, 3, 4};

    constexpr ParameterType                            ptype = ParameterType::send_counts;
    ContainerBasedConstBuffer<std::vector<int>, ptype> buffer_based_on_int_vector(int_vec);
    ContainerBasedConstBuffer<std::vector<int>, ptype> buffer_based_on_const_int_vector(int_vec_const);

    EXPECT_EQ(buffer_based_on_int_vector.get().size, int_vec.size());
    EXPECT_EQ(buffer_based_on_int_vector.get().ptr, int_vec.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_int_vector.get().ptr), const int*>);

    EXPECT_EQ(buffer_based_on_const_int_vector.get().size, int_vec_const.size());
    EXPECT_EQ(buffer_based_on_const_int_vector.get().ptr, int_vec_const.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_const_int_vector.get().ptr), const int*>);
}

TEST(Test_ContainerBasedConstBuffer, get_containers_other_than_vector) {
    std::string                                         str = "I am underlying storage";
    OwnContainer<int>                                   own_container;
    constexpr ParameterType                             ptype = ParameterType::send_counts;
    ContainerBasedConstBuffer<std::string, ptype>       buffer_based_on_string(str);
    ContainerBasedConstBuffer<OwnContainer<int>, ptype> buffer_based_on_own_container(own_container);

    EXPECT_EQ(buffer_based_on_string.get().size, str.size());
    EXPECT_EQ(buffer_based_on_string.get().ptr, str.data());

    EXPECT_EQ(buffer_based_on_own_container.get().size, own_container.size());
    EXPECT_EQ(buffer_based_on_own_container.get().ptr, own_container.data());
}

TEST(Test_UserAllocatedContainerBasedBuffer, get_ptr_basics) {
    std::vector<int> int_vec{1, 2, 3, 2, 1};

    constexpr ParameterType                                    ptype = ParameterType::send_counts;
    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype> buffer_based_on_int_vector(int_vec);

    auto resize_write_check = [&](size_t requested_size) {
        int* ptr = buffer_based_on_int_vector.get_ptr(requested_size);
        EXPECT_EQ(ptr, int_vec.data());
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
            EXPECT_EQ(ptr[i], int_vec[i]);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    resize_write_check(9);
}

TEST(Test_UserAllocatedContainerBasedBuffer, get_ptr_containers_other_than_vector) {
    OwnContainer<int> own_container;

    constexpr ParameterType                                     ptype = ParameterType::recv_counts;
    UserAllocatedContainerBasedBuffer<OwnContainer<int>, ptype> buffer_based_on_own_container(own_container);

    auto resize_write_check = [&](size_t requested_size) {
        int* ptr = buffer_based_on_own_container.get_ptr(requested_size);
        EXPECT_EQ(ptr, own_container.data());
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
            EXPECT_EQ(ptr[i], own_container[i]);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    resize_write_check(9);
}

TEST(Test_LibAllocatedContainerBasedBuffer, get_ptr_extract_basics) {
    constexpr ParameterType                                   ptype = ParameterType::recv_counts;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ptype> buffer_based_on_int_vector;

    auto resize_write_check = [&](size_t requested_size) {
        int* ptr = buffer_based_on_int_vector.get_ptr(requested_size);
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

TEST(Test_LibAllocatedContainerBasedBuffer, get_ptr_extract_containers_other_than_vector) {
    constexpr ParameterType                                    ptype = ParameterType::recv_counts;
    LibAllocatedContainerBasedBuffer<OwnContainer<int>, ptype> buffer_based_on_own_container;

    auto resize_write_check = [&](size_t requested_size) {
        int* ptr = buffer_based_on_own_container.get_ptr(requested_size);
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    const size_t last_resize = 9;
    resize_write_check(last_resize);
    OwnContainer<int> underlying_container = buffer_based_on_own_container.extract();
    for (size_t i = 0; i < last_resize; ++i) {
        EXPECT_EQ(underlying_container[i], static_cast<int>(last_resize - i));
    }
}
