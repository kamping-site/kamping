// This file is part of KaMPIng.
//
// Copyright 2021-2024 The KaMPIng Authors
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

#include <array>
#include <deque>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/assertion_levels.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"
#include "legacy_parameter_objects.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

TEST(HasDataMemberTest, has_data_member_basics) {
    EXPECT_TRUE(kamping::internal::has_data_member_v<std::vector<int>>);
    EXPECT_TRUE(kamping::internal::has_data_member_v<std::vector<double>>);
    EXPECT_TRUE((kamping::internal::has_data_member_v<std::vector<double, ::testing::CustomAllocator<double>>>));
    EXPECT_TRUE((kamping::internal::has_data_member_v<std::string>));
    EXPECT_TRUE((kamping::internal::has_data_member_v<std::array<int, 42>>));

    EXPECT_FALSE((kamping::internal::has_data_member_v<int>));
    EXPECT_FALSE((kamping::internal::has_data_member_v<bool>));

    // on some compilers vector<bool> still has .data() but it returns void
    // EXPECT_FALSE((kamping::internal::has_data_member_v<std::vector<bool>>));
    // EXPECT_FALSE((kamping::internal::has_data_member_v<std::vector<bool, testing::CustomAllocator<bool>>>));
}

TEST(IsSpecializationTest, is_specialization_basics) {
    EXPECT_TRUE((kamping::internal::is_specialization<std::vector<int>, std::vector>::value));
    EXPECT_TRUE((kamping::internal::is_specialization<std::vector<bool>, std::vector>::value));
    EXPECT_TRUE(
        (kamping::internal::is_specialization<std::vector<int, ::testing::CustomAllocator<int>>, std::vector>::value)
    );
    EXPECT_TRUE((kamping::internal::
                     is_specialization<std::vector<double, ::testing::CustomAllocator<double>>, std::vector>::value));
    EXPECT_TRUE((kamping::internal::is_specialization<std::deque<int>, std::deque>::value));

    EXPECT_FALSE((kamping::internal::is_specialization<std::array<int, 2>, std::vector>::value));
    EXPECT_FALSE((kamping::internal::is_specialization<std::deque<int>, std::vector>::value));
    EXPECT_FALSE((kamping::internal::is_specialization<int, std::vector>::value));
}
TEST(HasValueTypeTest, has_value_type_basics) {
    EXPECT_TRUE(kamping::internal::has_value_type_v<std::vector<int>>);
    EXPECT_TRUE(kamping::internal::has_value_type_v<std::vector<bool>>);
    EXPECT_TRUE((kamping::internal::has_value_type_v<std::array<int, 42>>));
    EXPECT_TRUE((kamping::internal::has_value_type_v<std::string>));

    EXPECT_FALSE((kamping::internal::has_value_type_v<int>));
    EXPECT_FALSE((kamping::internal::has_value_type_v<double>));
    EXPECT_FALSE((kamping::internal::has_value_type_v<bool>));
}

TEST(IsVectorBoolTest, is_vector_bool_basics) {
    EXPECT_TRUE(kamping::internal::is_vector_bool_v<std::vector<bool>>);
    EXPECT_TRUE((kamping::internal::is_vector_bool_v<std::vector<bool, ::testing::CustomAllocator<bool>>>));
    EXPECT_TRUE(kamping::internal::is_vector_bool_v<std::vector<bool> const>);
    EXPECT_TRUE(kamping::internal::is_vector_bool_v<std::vector<bool>&>);
    EXPECT_TRUE(kamping::internal::is_vector_bool_v<std::vector<bool> const&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<std::vector<int>>);
    EXPECT_FALSE((kamping::internal::is_vector_bool_v<std::vector<int, ::testing::CustomAllocator<int>>>));
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<std::vector<int>&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<std::vector<int> const&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<std::vector<kamping::kabool>>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<std::vector<kamping::kabool>&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<std::vector<kamping::kabool> const&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<bool>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<bool&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<bool const&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<int>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<int&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<int const&>);
}

// Tests the basic functionality of EmptyBuffer
TEST(EmptyBufferTest, get_basics) {
    constexpr ParameterType                                                         ptype = ParameterType::send_counts;
    EmptyDataBuffer<std::vector<int>, ptype, kamping::internal::BufferType::ignore> empty_buffer{};

    EXPECT_EQ(empty_buffer.size(), 0);
    EXPECT_EQ(empty_buffer.get().size(), 0);
    EXPECT_EQ(empty_buffer.get().data(), nullptr);
    EXPECT_EQ(empty_buffer.data(), nullptr);
}

// Tests the basic functionality of ContainerBasedConstBuffer
TEST(ContainerBasedConstBufferTest, get_basics) {
    std::vector<int>       int_vec{1, 2, 3};
    std::vector<int> const int_vec_const{1, 2, 3, 4};

    constexpr ParameterType                                   ptype = ParameterType::send_counts;
    constexpr BufferType                                      btype = BufferType::in_buffer;
    ContainerBasedConstBuffer<std::vector<int>, ptype, btype> buffer_based_on_int_vector(int_vec);
    ContainerBasedConstBuffer<std::vector<int>, ptype, btype> buffer_based_on_const_int_vector(int_vec_const);

    EXPECT_EQ(buffer_based_on_int_vector.size(), int_vec.size());
    EXPECT_EQ(buffer_based_on_int_vector.get().size(), int_vec.size());
    EXPECT_EQ(buffer_based_on_int_vector.get().data(), int_vec.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_int_vector.get().data()), int const*>);
    EXPECT_EQ(buffer_based_on_int_vector.data(), int_vec.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_int_vector.data()), int const*>);
    EXPECT_FALSE(decltype(buffer_based_on_int_vector)::is_out_buffer);
    EXPECT_FALSE(decltype(buffer_based_on_int_vector)::is_lib_allocated);

    EXPECT_EQ(buffer_based_on_const_int_vector.get().size(), int_vec_const.size());
    EXPECT_EQ(buffer_based_on_const_int_vector.get().data(), int_vec_const.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_const_int_vector.get().data()), int const*>);
    EXPECT_EQ(buffer_based_on_const_int_vector.data(), int_vec_const.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_const_int_vector.data()), int const*>);
    EXPECT_FALSE(decltype(buffer_based_on_const_int_vector)::is_out_buffer);
    EXPECT_FALSE(decltype(buffer_based_on_const_int_vector)::is_lib_allocated);
}

TEST(ContainerBasedConstBufferTest, get_containers_other_than_vector) {
    std::string                                                           str = "I am underlying storage";
    ::testing::OwnContainer<int>                                          own_container;
    constexpr ParameterType                                               ptype = ParameterType::send_buf;
    constexpr BufferType                                                  btype = BufferType::in_buffer;
    ContainerBasedConstBuffer<std::string, ptype, btype>                  buffer_based_on_string(str);
    ContainerBasedConstBuffer<::testing::OwnContainer<int>, ptype, btype> buffer_based_on_own_container(own_container);

    EXPECT_EQ(buffer_based_on_string.get().size(), str.size());
    EXPECT_EQ(buffer_based_on_string.get().data(), str.data());

    EXPECT_EQ(buffer_based_on_own_container.get().size(), own_container.size());
    EXPECT_EQ(buffer_based_on_own_container.get().data(), own_container.data());
}

TEST(ContainerBasedConstBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                                   ptype = ParameterType::send_counts;
    constexpr BufferType                                      btype = BufferType::in_buffer;
    std::vector<int> const                                    container{1, 2, 3};
    ContainerBasedConstBuffer<std::vector<int>, ptype, btype> buffer1(container);
    ContainerBasedConstBuffer<std::vector<int>, ptype, btype> buffer2(std::move(buffer1));
    EXPECT_EQ(buffer2.get().size(), container.size());
    EXPECT_TRUE(std::equal(container.begin(), container.end(), buffer2.get().data()));
}

// Tests the basic functionality of ContainerBasedOwningBuffer
TEST(ContainerBasedOwningBufferTest, get_basics) {
    std::vector<int> int_vec{1, 2, 3};

    constexpr ParameterType                                    ptype = ParameterType::send_counts;
    constexpr BufferType                                       btype = BufferType::in_buffer;
    ContainerBasedOwningBuffer<std::vector<int>, ptype, btype> buffer_based_on_moved_vector(std::move(int_vec));
    ContainerBasedOwningBuffer<std::vector<int>, ptype, btype> buffer_based_on_rvalue_vector(std::vector<int>{1, 2, 3});

    EXPECT_EQ(buffer_based_on_moved_vector.size(), 3);
    EXPECT_EQ(buffer_based_on_moved_vector.get().size(), 3);
    EXPECT_EQ(buffer_based_on_moved_vector.get().data()[0], 1);
    EXPECT_EQ(buffer_based_on_moved_vector.get().data()[1], 2);
    EXPECT_EQ(buffer_based_on_moved_vector.get().data()[2], 3);
    static_assert(std::is_same_v<decltype(buffer_based_on_moved_vector.get().data()), int const*>);
    EXPECT_EQ(buffer_based_on_moved_vector.data()[0], 1);
    EXPECT_EQ(buffer_based_on_moved_vector.data()[1], 2);
    EXPECT_EQ(buffer_based_on_moved_vector.data()[2], 3);
    static_assert(std::is_same_v<decltype(buffer_based_on_moved_vector.data()), int const*>);
    EXPECT_FALSE(decltype(buffer_based_on_moved_vector)::is_out_buffer);
    EXPECT_FALSE(decltype(buffer_based_on_moved_vector)::is_lib_allocated);

    EXPECT_EQ(buffer_based_on_rvalue_vector.size(), 3);
    EXPECT_EQ(buffer_based_on_rvalue_vector.get().size(), 3);
    EXPECT_EQ(buffer_based_on_rvalue_vector.get().data()[0], 1);
    EXPECT_EQ(buffer_based_on_rvalue_vector.get().data()[1], 2);
    EXPECT_EQ(buffer_based_on_rvalue_vector.get().data()[2], 3);
    static_assert(std::is_same_v<decltype(buffer_based_on_rvalue_vector.get().data()), int const*>);
    EXPECT_EQ(buffer_based_on_rvalue_vector.data()[0], 1);
    EXPECT_EQ(buffer_based_on_rvalue_vector.data()[1], 2);
    EXPECT_EQ(buffer_based_on_rvalue_vector.data()[2], 3);
    static_assert(std::is_same_v<decltype(buffer_based_on_rvalue_vector.data()), int const*>);
    EXPECT_FALSE(decltype(buffer_based_on_rvalue_vector)::is_out_buffer);
    EXPECT_FALSE(decltype(buffer_based_on_rvalue_vector)::is_lib_allocated);

    {
        auto const& underlying_container = buffer_based_on_moved_vector.underlying();
        EXPECT_EQ(underlying_container, (std::vector<int>{1, 2, 3}));
    }
    {
        auto const& underlying_container = buffer_based_on_rvalue_vector.underlying();
        EXPECT_EQ(underlying_container, (std::vector<int>{1, 2, 3}));
    }
}

TEST(ContainerBasedOwningBufferTest, get_containers_other_than_vector) {
    constexpr ParameterType ptype = ParameterType::send_buf;
    constexpr BufferType    btype = BufferType::in_buffer;

    // string
    std::string                                           str      = "I am underlying storage";
    std::string                                           expected = "I am underlying storage";
    ContainerBasedOwningBuffer<std::string, ptype, btype> buffer_based_on_string(std::move(str));

    EXPECT_EQ(buffer_based_on_string.get().size(), expected.size());
    EXPECT_EQ(
        std::string(
            buffer_based_on_string.get().data(),
            buffer_based_on_string.get().data() + buffer_based_on_string.get().size()
        ),
        expected
    );
    {
        auto const& underlying_container = buffer_based_on_string.underlying();
        EXPECT_EQ(underlying_container, expected);
    }
    // own container
    ::testing::OwnContainer<int> own_container{1, 2, 3};
    EXPECT_EQ(own_container.copy_count(), 0);

    ContainerBasedOwningBuffer<::testing::OwnContainer<int>, ptype, btype> buffer_based_on_own_container(
        std::move(own_container)
    );
    EXPECT_EQ(own_container.copy_count(), 0);
    EXPECT_EQ(buffer_based_on_own_container.underlying().copy_count(), 0);

    EXPECT_EQ(buffer_based_on_own_container.get().size(), 3);
    EXPECT_EQ(buffer_based_on_own_container.get().data()[0], 1);
    EXPECT_EQ(buffer_based_on_own_container.get().data()[1], 2);
    EXPECT_EQ(buffer_based_on_own_container.get().data()[2], 3);
    {
        auto const& underlying_container = buffer_based_on_own_container.underlying();
        EXPECT_EQ(underlying_container, (::testing::OwnContainer<int>{1, 2, 3}));
    }
}

TEST(ContainerBasedOwningBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                                    ptype = ParameterType::send_counts;
    constexpr BufferType                                       btype = BufferType::in_buffer;
    std::vector<int> const                                     container{1, 2, 3};
    ContainerBasedOwningBuffer<std::vector<int>, ptype, btype> buffer1({1, 2, 3});
    ContainerBasedOwningBuffer<std::vector<int>, ptype, btype> buffer2(std::move(buffer1));
    EXPECT_EQ(buffer2.get().size(), 3);

    std::vector<int> const expected_container{1, 2, 3};
    EXPECT_TRUE(std::equal(expected_container.begin(), expected_container.end(), buffer2.get().data()));
}

TEST(UserAllocatedContainerBasedBufferTest, resize_and_data_basics) {
    std::vector<int> int_vec{1, 2, 3, 2, 1};

    constexpr ParameterType      ptype         = ParameterType::send_counts;
    constexpr BufferType         btype         = BufferType::in_buffer;
    constexpr BufferResizePolicy resize_policy = BufferResizePolicy::resize_to_fit;
    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, resize_policy> buffer_based_on_int_vector(int_vec
    );
    EXPECT_EQ(int_vec.size(), buffer_based_on_int_vector.get().size());
    EXPECT_EQ(int_vec.data(), buffer_based_on_int_vector.get().data());
    EXPECT_FALSE(decltype(buffer_based_on_int_vector)::is_out_buffer);
    EXPECT_FALSE(decltype(buffer_based_on_int_vector)::is_lib_allocated);

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
    ::testing::OwnContainer<int> own_container;

    constexpr ParameterType      ptype         = ParameterType::recv_counts;
    constexpr BufferType         btype         = BufferType::in_buffer;
    constexpr BufferResizePolicy resize_policy = BufferResizePolicy::resize_to_fit;
    UserAllocatedContainerBasedBuffer<::testing::OwnContainer<int>, ptype, btype, resize_policy>
        buffer_based_on_own_container(own_container);

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

TEST(UserAllocatedContainerBasedBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType      ptype = ParameterType::send_counts;
    constexpr BufferType         btype = BufferType::in_buffer;
    std::vector<int>             container{1, 2, 3};
    auto const                   const_container = container; // ensure that container is not altered
    constexpr BufferResizePolicy resize_policy   = BufferResizePolicy::resize_to_fit;

    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, resize_policy> buffer1(container);
    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, resize_policy> buffer2(std::move(buffer1));
    EXPECT_EQ(buffer2.get().size(), const_container.size());
    EXPECT_TRUE(std::equal(const_container.begin(), const_container.end(), buffer2.get().data()));
}

TEST(LibAllocatedContainerBasedBufferTest, resize_and_data_extract_basics) {
    constexpr ParameterType                                          ptype = ParameterType::recv_counts;
    constexpr BufferType                                             btype = BufferType::in_buffer;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype> buffer_based_on_int_vector;
    EXPECT_FALSE(decltype(buffer_based_on_int_vector)::is_out_buffer);
    EXPECT_TRUE(decltype(buffer_based_on_int_vector)::is_lib_allocated);

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
    size_t const last_resize = 9;
    resize_write_check(last_resize);

    // The buffer will be in an invalid state after extraction; that's why we have to access these attributes
    // beforehand.
    auto const       size_of_buffer        = buffer_based_on_int_vector.size();
    auto const       data_of_buffer        = buffer_based_on_int_vector.data();
    auto const       size_of_get_of_buffer = buffer_based_on_int_vector.get().size();
    auto const       data_of_get_of_buffer = buffer_based_on_int_vector.get().data();
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
    constexpr ParameterType                                                      ptype = ParameterType::recv_counts;
    constexpr BufferType                                                         btype = BufferType::in_buffer;
    LibAllocatedContainerBasedBuffer<::testing::OwnContainer<int>, ptype, btype> buffer_based_on_own_container;

    auto resize_write_check = [&](size_t requested_size) {
        buffer_based_on_own_container.resize(requested_size);
        int* ptr = buffer_based_on_own_container.data();
        for (size_t i = 0; i < requested_size; ++i) {
            ptr[i] = static_cast<int>(requested_size - i);
        }
    };
    resize_write_check(10);
    resize_write_check(50);
    size_t const last_resize = 9;
    resize_write_check(last_resize);
    ::testing::OwnContainer<int> underlying_container = buffer_based_on_own_container.extract();
    for (size_t i = 0; i < last_resize; ++i) {
        EXPECT_EQ(underlying_container[i], static_cast<int>(last_resize - i));
    }
}

TEST(LibAllocatedContainerBasedBufferTest, move_ctor_assignment_operator_is_enabled) {
    constexpr ParameterType                                                      ptype = ParameterType::recv_counts;
    constexpr BufferType                                                         btype = BufferType::in_buffer;
    LibAllocatedContainerBasedBuffer<::testing::OwnContainer<int>, ptype, btype> buffer1;
    size_t const                                                                 size = 3;
    buffer1.resize(size);
    buffer1.get().data()[0] = 0;
    buffer1.get().data()[1] = 1;
    buffer1.get().data()[2] = 2;
    EXPECT_EQ(decltype(buffer1)::parameter_type, ptype);
    LibAllocatedContainerBasedBuffer<::testing::OwnContainer<int>, ptype, btype> buffer2(std::move(buffer1));
    EXPECT_EQ(decltype(buffer2)::parameter_type, ptype);
    LibAllocatedContainerBasedBuffer<::testing::OwnContainer<int>, ptype, btype> buffer3;
    buffer3 = std::move(buffer2);
    EXPECT_EQ(buffer3.get().size(), 3);
    EXPECT_EQ(buffer3.get().data()[0], 0);
    EXPECT_EQ(buffer3.get().data()[1], 1);
    EXPECT_EQ(buffer3.get().data()[2], 2);
    EXPECT_EQ(decltype(buffer3)::parameter_type, ptype);
}

TEST(SingleElementConstBufferTest, get_basics) {
    constexpr ParameterType                     ptype = ParameterType::send_counts;
    constexpr BufferType                        btype = BufferType::in_buffer;
    int                                         value = 5;
    SingleElementConstBuffer<int, ptype, btype> int_buffer(value);
    EXPECT_FALSE(decltype(int_buffer)::is_out_buffer);
    EXPECT_FALSE(decltype(int_buffer)::is_lib_allocated);

    EXPECT_EQ(int_buffer.size(), 1);
    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);
    EXPECT_EQ(*(int_buffer.data()), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_FALSE(int_buffer.is_modifiable);
    EXPECT_FALSE(int_buffer.is_out_buffer);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, decltype(value)>);
}

TEST(SingleElementConstBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                     ptype = ParameterType::send_counts;
    constexpr BufferType                        btype = BufferType::in_buffer;
    int const                                   elem  = 42;
    SingleElementConstBuffer<int, ptype, btype> buffer1(elem);
    SingleElementConstBuffer<int, ptype, btype> buffer2(std::move(buffer1));
    EXPECT_EQ(*buffer2.get().data(), elem);
    EXPECT_EQ(*buffer2.data(), elem);
    EXPECT_EQ(buffer2.get_single_element(), elem);
}

TEST(SingleElementOwningBufferTest, get_basics) {
    constexpr ParameterType                      ptype = ParameterType::send_counts;
    constexpr BufferType                         btype = BufferType::in_buffer;
    SingleElementOwningBuffer<int, ptype, btype> int_buffer(5);
    EXPECT_FALSE(decltype(int_buffer)::is_out_buffer);
    EXPECT_FALSE(decltype(int_buffer)::is_lib_allocated);

    EXPECT_EQ(int_buffer.size(), 1);
    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);
    EXPECT_EQ(*(int_buffer.data()), 5);
    EXPECT_EQ(int_buffer.underlying(), 5);
    EXPECT_EQ(int_buffer.get_single_element(), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_FALSE(int_buffer.is_modifiable);
    EXPECT_FALSE(int_buffer.is_out_buffer);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, int>);
}

TEST(SingleElementOwningBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                      ptype = ParameterType::send_counts;
    constexpr BufferType                         btype = BufferType::in_buffer;
    SingleElementOwningBuffer<int, ptype, btype> buffer1(42);
    SingleElementOwningBuffer<int, ptype, btype> buffer2(std::move(buffer1));
    EXPECT_EQ(*buffer2.get().data(), 42);
    EXPECT_EQ(*buffer2.data(), 42);
    EXPECT_EQ(buffer2.underlying(), 42);
    EXPECT_EQ(buffer2.get_single_element(), 42);
}

TEST(SingleElementModifiableBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                          ptype      = ParameterType::send_counts;
    constexpr BufferType                             btype      = BufferType::in_buffer;
    int                                              elem       = 42;
    int const                                        const_elem = elem;
    SingleElementModifiableBuffer<int, ptype, btype> buffer1(elem);
    SingleElementModifiableBuffer<int, ptype, btype> buffer2(std::move(buffer1));
    EXPECT_FALSE(decltype(buffer1)::is_out_buffer);
    EXPECT_FALSE(decltype(buffer1)::is_lib_allocated);
    EXPECT_FALSE(decltype(buffer2)::is_out_buffer);
    EXPECT_FALSE(decltype(buffer2)::is_lib_allocated);
    EXPECT_EQ(*buffer2.get().data(), const_elem);
    EXPECT_EQ(*buffer2.data(), const_elem);
    EXPECT_EQ(buffer2.get_single_element(), const_elem);
}

TEST(SingleElementModifiableBufferTest, get_basics) {
    constexpr ParameterType                          ptype = ParameterType::send_counts;
    constexpr BufferType                             btype = BufferType::in_buffer;
    int                                              value = 5;
    SingleElementModifiableBuffer<int, ptype, btype> int_buffer(value);

    EXPECT_EQ(int_buffer.size(), 1);

    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);
    EXPECT_EQ(*(int_buffer.data()), 5);
    EXPECT_EQ(int_buffer.get_single_element(), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_TRUE(int_buffer.is_modifiable);
    EXPECT_FALSE(int_buffer.is_out_buffer);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, decltype(value)>);
}

TEST(LibAllocatedSingleElementBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                            ptype      = ParameterType::send_counts;
    constexpr BufferType                               btype      = BufferType::in_buffer;
    int                                                elem       = 42;
    int const                                          const_elem = elem;
    LibAllocatedSingleElementBuffer<int, ptype, btype> buffer1{};
    *buffer1.get().data() = elem;
    LibAllocatedSingleElementBuffer<int, ptype, btype> buffer2(std::move(buffer1));
    EXPECT_EQ(*buffer2.get().data(), const_elem);
    EXPECT_EQ(*buffer2.data(), const_elem);
    EXPECT_EQ(buffer2.get_single_element(), const_elem);
    EXPECT_FALSE(decltype(buffer1)::is_out_buffer);
    EXPECT_TRUE(decltype(buffer1)::is_lib_allocated);
    EXPECT_FALSE(decltype(buffer2)::is_out_buffer);
    EXPECT_TRUE(decltype(buffer2)::is_lib_allocated);
}

TEST(LibAllocatedSingleElementBufferTest, get_basics) {
    constexpr ParameterType                            ptype = ParameterType::send_counts;
    constexpr BufferType                               btype = BufferType::in_buffer;
    int                                                value = 5;
    LibAllocatedSingleElementBuffer<int, ptype, btype> int_buffer{};

    *int_buffer.get().data() = value;

    EXPECT_EQ(int_buffer.size(), 1);

    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);
    EXPECT_EQ(*(int_buffer.data()), 5);
    EXPECT_EQ(int_buffer.get_single_element(), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_TRUE(int_buffer.is_modifiable);
    EXPECT_FALSE(int_buffer.is_out_buffer);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, decltype(value)>);

    int extracted_value = int_buffer.extract();
    EXPECT_EQ(extracted_value, value);
}

TEST(RootTest, move_constructor_assignment_operator_is_enabled) {
    int            rank       = 2;
    int const      const_rank = rank;
    RootDataBuffer root1(rank);
    RootDataBuffer root2 = std::move(root1);
    RootDataBuffer root3(rank + 1);
    root3 = std::move(root2);
    EXPECT_EQ(root3.rank_signed(), const_rank);
}

TEST(UserAllocatedContainerBasedBufferTest, resize_user_allocated_buffer) {
    std::vector<int>        data(20, 0);
    Span<int>               container = {data.data(), data.size()};
    constexpr ParameterType ptype     = ParameterType::send_counts;
    constexpr BufferType    btype     = BufferType::in_buffer;

    UserAllocatedContainerBasedBuffer<Span<int>, ptype, btype, no_resize> span_buffer(container);

    for (size_t i = 0; i <= 20; ++i) {
        bool resize_called = false;
        span_buffer.resize_if_requested([&] {
            resize_called = true;
            return i;
        });
        EXPECT_FALSE(resize_called);
        EXPECT_EQ(20, span_buffer.size());
    }

    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, resize_to_fit> vec_buffer(data);

    for (size_t i = 0; i <= 20; ++i) {
        vec_buffer.resize(i);
        EXPECT_EQ(i, vec_buffer.size());
    }
}

TEST(DataBufferTest, has_extract) {
    static_assert(
        has_extract_v<DataBuffer<
            int,
            ParameterType,
            ParameterType::send_buf,
            BufferModifiability::modifiable,
            BufferOwnership::owning,
            BufferType::in_buffer,
            BufferResizePolicy::no_resize,
            BufferAllocation::lib_allocated>>,
        "Library allocated DataBuffers must have an extract() member function"
    );
    static_assert(
        has_extract_v<DataBuffer<
            int,
            ParameterType,
            ParameterType::send_buf,
            BufferModifiability::modifiable,
            BufferOwnership::owning,
            BufferType::in_buffer,
            BufferResizePolicy::no_resize,
            BufferAllocation::user_allocated>>,
        "User allocated owning DataBuffers must have an extract() member function"
    );
    static_assert(
        !has_extract_v<DataBuffer<
            int,
            ParameterType,
            ParameterType::send_buf,
            BufferModifiability::modifiable,
            BufferOwnership::referencing,
            BufferType::in_buffer,
            BufferResizePolicy::no_resize,
            BufferAllocation::user_allocated>>,
        "User allocated referencing DataBuffers must not have an extract() member function"
    );
}

TEST(DataBufferTest, resize_if_requested_with_resize_to_fit) {
    std::vector<int>        data;
    constexpr ParameterType ptype = ParameterType::send_counts;
    constexpr BufferType    btype = BufferType::in_buffer;

    size_t const required_size = 42;
    int          call_counter  = 0;
    auto         size_function = [&call_counter]() {
        ++call_counter;
        return required_size;
    };
    {
        UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, BufferResizePolicy::resize_to_fit> buffer(data
        );
        buffer.resize_if_requested(size_function);
        EXPECT_TRUE((has_member_resize_v<decltype(buffer), size_t>));
        EXPECT_EQ(call_counter, 1);
        EXPECT_EQ(data.size(), required_size);
    }
    // reset call counter
    call_counter = 0;
    {
        data.resize(2 * required_size);
        UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, BufferResizePolicy::resize_to_fit> buffer(data
        );
        buffer.resize_if_requested(size_function);
        EXPECT_TRUE((has_member_resize_v<decltype(buffer), size_t>));
        EXPECT_EQ(call_counter, 1);
        EXPECT_EQ(data.size(), required_size);
    }
}

TEST(DataBufferTest, resize_if_requested_with_grow_only) {
    std::vector<int>        data;
    constexpr ParameterType ptype = ParameterType::send_counts;
    constexpr BufferType    btype = BufferType::in_buffer;

    size_t const required_size = 42;
    int          call_counter  = 0;
    auto         size_function = [&call_counter]() {
        ++call_counter;
        return required_size;
    };
    {
        UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, BufferResizePolicy::grow_only> buffer(data);
        buffer.resize_if_requested(size_function);
        EXPECT_TRUE((has_member_resize_v<decltype(buffer), size_t>));
        EXPECT_EQ(call_counter, 1);
        EXPECT_EQ(data.size(), required_size);
    }
    // reset call counter
    call_counter = 0;
    {
        data.resize(2 * required_size);
        UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, BufferResizePolicy::grow_only> buffer(data);
        buffer.resize_if_requested(size_function);
        EXPECT_TRUE((has_member_resize_v<decltype(buffer), size_t>));
        EXPECT_EQ(call_counter, 1);
        EXPECT_EQ(data.size(), 2 * required_size);
    }
}

TEST(DataBufferTest, resize_if_requested_with_no_resize) {
    std::vector<int>        data;
    constexpr ParameterType ptype = ParameterType::send_counts;
    constexpr BufferType    btype = BufferType::in_buffer;

    size_t const required_size = 42;
    int          call_counter  = 0;
    auto         size_function = [&call_counter]() {
        ++call_counter;
        return required_size;
    };
    {
        UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, BufferResizePolicy::no_resize> buffer(data);
        buffer.resize_if_requested(size_function);
        EXPECT_FALSE((has_member_resize_v<decltype(buffer), size_t>));
        EXPECT_EQ(call_counter, 0);
        EXPECT_EQ(data.size(), 0);
    }
    // reset call counter
    call_counter = 0;
    {
        data.resize(2 * required_size);
        UserAllocatedContainerBasedBuffer<std::vector<int>, ptype, btype, BufferResizePolicy::no_resize> buffer(data);
        buffer.resize_if_requested(size_function);
        EXPECT_FALSE((has_member_resize_v<decltype(buffer), size_t>));
        EXPECT_EQ(call_counter, 0);
        EXPECT_EQ(data.size(), 2 * required_size);
    }
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST(LibAllocatedContainerBasedBufferTest, prevent_usage_after_extraction) {
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_buf, BufferType::in_buffer> buffer;

    buffer.data();
    buffer.size();
    buffer.resize(10);
    std::ignore = buffer.extract();
    EXPECT_KASSERT_FAILS(buffer.extract(), "Cannot extract a buffer that has already been extracted.");
    EXPECT_KASSERT_FAILS(buffer.get(), "Cannot get a buffer that has already been extracted.");
    EXPECT_KASSERT_FAILS(buffer.data(), "Cannot get a pointer to a buffer that has already been extracted.");
    EXPECT_KASSERT_FAILS(buffer.size(), "Cannot get the size of a buffer that has already been extracted.");
    EXPECT_KASSERT_FAILS(buffer.resize(20), "Cannot resize a buffer that has already been extracted.");
}

TEST(LibAllocatedContainerBasedBufferTest, prevent_usage_after_extraction_via_mpi_result) {
    using OutParams = std::tuple<
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_buf, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_counts, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs, BufferType::out_buffer>,
        // we use out_buffer here because extracting is only done from out buffers
        LibAllocatedContainerBasedBuffer<int, ParameterType::recv_count, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<int, ParameterType::send_count, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<int, ParameterType::send_recv_count, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_type, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::recv_type, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<MPI_Datatype, ParameterType::send_recv_type, BufferType::out_buffer>,
        LibAllocatedContainerBasedBuffer<Status, ParameterType::status, BufferType::out_buffer>>;

    std::tuple_element_t<0, OutParams> recv_buffer;
    std::tuple_element_t<1, OutParams> recv_counts;
    std::tuple_element_t<2, OutParams> recv_displs;
    std::tuple_element_t<3, OutParams> send_counts;
    std::tuple_element_t<4, OutParams> send_displs;
    // we use out_buffer here because extracting is only done from out buffers
    std::tuple_element_t<5, OutParams>  recv_count;
    std::tuple_element_t<6, OutParams>  send_count;
    std::tuple_element_t<7, OutParams>  send_recv_count;
    std::tuple_element_t<8, OutParams>  send_type;
    std::tuple_element_t<9, OutParams>  recv_type;
    std::tuple_element_t<10, OutParams> send_recv_type;
    std::tuple_element_t<11, OutParams> status;

    MPIResult result = make_mpi_result<OutParams>(
        std::move(status),
        std::move(recv_buffer),
        std::move(recv_counts),
        std::move(recv_count),
        std::move(recv_displs),
        std::move(send_counts),
        std::move(send_count),
        std::move(send_displs),
        std::move(send_recv_count),
        std::move(send_type),
        std::move(recv_type),
        std::move(send_recv_type)
    );

    std::ignore = result.extract_status();
    EXPECT_KASSERT_FAILS(result.extract_status(), "Cannot extract a status that has already been extracted.");

    std::ignore = result.extract_recv_buffer();
    EXPECT_KASSERT_FAILS(result.extract_recv_buffer(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_recv_counts();
    EXPECT_KASSERT_FAILS(result.extract_recv_counts(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_recv_displs();
    EXPECT_KASSERT_FAILS(result.extract_recv_displs(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_send_counts();
    EXPECT_KASSERT_FAILS(result.extract_send_counts(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_send_displs();
    EXPECT_KASSERT_FAILS(result.extract_send_displs(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_recv_count();
    EXPECT_KASSERT_FAILS(result.extract_recv_count(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_send_count();
    EXPECT_KASSERT_FAILS(result.extract_send_count(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_send_recv_count();
    EXPECT_KASSERT_FAILS(result.extract_send_recv_count(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_send_type();
    EXPECT_KASSERT_FAILS(result.extract_send_type(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_recv_type();
    EXPECT_KASSERT_FAILS(result.extract_recv_type(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_send_recv_type();
    EXPECT_KASSERT_FAILS(result.extract_send_recv_type(), "Cannot extract a buffer that has already been extracted.");
}
#endif

TEST(DataBufferTest, make_data_buffer) {
    {
        // Constant, container, referencing, user allocated
        std::vector<int>                  vec;
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::constant,
            btype,
            BufferResizePolicy::no_resize>(vec);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        // As this buffer is referencing, the addresses of vec ad data_buf.underlying() should be the same.
        EXPECT_EQ(&vec, &data_buf.underlying());
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int> const&>,
            "Referencing buffers must hold a reference to their data."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, container, referencing, user allocated
        std::vector<int>                  vec;
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::modifiable,
            btype,
            BufferResizePolicy::grow_only>(vec);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::grow_only);
        // As this buffer is referencing, the addresses of vec ad data_buf.underlying() should be the same.
        EXPECT_EQ(&vec, &data_buf.underlying());
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int>&>,
            "Referencing buffers must hold a reference to their data."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, single element, referencing, user allocated
        int                               single_int;
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::constant,
            btype,
            BufferResizePolicy::no_resize>(single_int);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_TRUE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        // As this buffer is referencing, the addresses of vec ad data_buf.underlying() should be the same.
        EXPECT_EQ(&single_int, &data_buf.underlying());
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, int const&>,
            "Referencing buffers must hold a reference to their data."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, container, owning, user allocated
        std::vector<int>                  vec;
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::constant,
            btype,
            BufferResizePolicy::no_resize>(std::move(vec));
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int> const>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }

    {
        // modifiable, container, owning, library allocated
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::modifiable,
            btype,
            BufferResizePolicy::grow_only>(alloc_new<std::vector<int>>);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::grow_only);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int>>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, single element, owning, lib_allocated
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::modifiable,
            btype,
            BufferResizePolicy::no_resize>(alloc_new<int>);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_TRUE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, int>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, container, owning, user_allocated with initializer_list
        constexpr internal::ParameterType ptype         = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype         = internal::BufferType::in_buffer;
        constexpr BufferResizePolicy      resize_policy = BufferResizePolicy::no_resize;
        auto                              data_buf =
            internal::make_data_buffer_builder<ptype, BufferModifiability::modifiable, btype, resize_policy>({1, 2, 3})
                .construct_buffer_or_rebind();
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, resize_policy);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int>>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, container, owning, user_allocated with initializer_list
        constexpr internal::ParameterType ptype         = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype         = internal::BufferType::in_buffer;
        constexpr BufferResizePolicy      resize_policy = BufferResizePolicy::no_resize;
        auto                              data_buf =
            internal::make_data_buffer_builder<ptype, BufferModifiability::constant, btype, resize_policy>({1, 2, 3})
                .construct_buffer_or_rebind();
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, resize_policy);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int> const>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
}

TEST(DataBufferTest, make_data_buffer_boolean_value) {
    // use a custom container, because std::vector<bool> is not supported (see compilation failure tests)
    {
        // Constant, container, referencing, user allocated
        ::testing::OwnContainer<bool>     vec           = {true, false};
        constexpr internal::ParameterType ptype         = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype         = internal::BufferType::in_buffer;
        constexpr BufferResizePolicy      resize_policy = BufferResizePolicy::no_resize;
        auto                              data_buf      = internal::
            make_data_buffer<internal::ParameterType, ptype, BufferModifiability::constant, btype, resize_policy>(vec);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, resize_policy);
        // As this buffer is referencing, the addresses of vec ad data_buf.underlying() should be the same.
        EXPECT_EQ(&vec, &data_buf.underlying());
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, ::testing::OwnContainer<bool> const&>,
            "Referencing buffers must hold a reference to their data."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, container, referencing, user allocated
        ::testing::OwnContainer<bool>     vec      = {true, false};
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::modifiable,
            btype,
            BufferResizePolicy::resize_to_fit>(vec);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::resize_to_fit);
        // As this buffer is referencing, the addresses of vec ad data_buf.underlying() should be the same.
        EXPECT_EQ(&vec, &data_buf.underlying());
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, ::testing::OwnContainer<bool>&>,
            "Referencing buffers must hold a reference to their data."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, single element, referencing, user allocated
        bool                              single_bool;
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::constant,
            btype,
            BufferResizePolicy::no_resize>(single_bool);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_TRUE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        // As this buffer is referencing, the addresses of vec ad data_buf.underlying() should be the same.
        EXPECT_EQ(&single_bool, &data_buf.underlying());
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, bool const&>,
            "Referencing buffers must hold a reference to their data."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, container, owning, user allocated
        ::testing::OwnContainer<bool>     vec      = {true, false};
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::constant,
            btype,
            BufferResizePolicy::no_resize>(std::move(vec));
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, ::testing::OwnContainer<bool> const>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }

    {
        // modifiable, container, owning, library allocated
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::modifiable,
            btype,
            BufferResizePolicy::resize_to_fit>(alloc_new<::testing::OwnContainer<bool>>);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::resize_to_fit);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, ::testing::OwnContainer<bool>>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, single element, owning, lib_allocated
        constexpr internal::ParameterType ptype    = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype    = internal::BufferType::in_buffer;
        auto                              data_buf = internal::make_data_buffer<
            internal::ParameterType,
            ptype,
            BufferModifiability::modifiable,
            btype,
            BufferResizePolicy::no_resize>(alloc_new<bool>);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_TRUE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, bool>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, container, owning, user_allocated with initializer_list
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto                              data_buf =
            internal::
                make_data_buffer_builder<ptype, BufferModifiability::modifiable, btype, BufferResizePolicy::no_resize>(
                    {true, false, true}
                )
                    .construct_buffer_or_rebind();
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<kabool>>,
            "Initializer lists of type bool have to be converted to std::vector<kabool>."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, container, owning, user_allocated with initializer_list
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto                              data_buf =
            internal::
                make_data_buffer_builder<ptype, BufferModifiability::constant, btype, BufferResizePolicy::no_resize>(
                    {true, false, true}
                )
                    .construct_buffer_or_rebind();
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        EXPECT_EQ(data_buf.resize_policy, BufferResizePolicy::no_resize);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<kabool> const>,
            "Initializer lists of type bool have to be converted to std::vector<kabool>."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
}

TEST(DataBufferTest, referencing_buffers_are_copyable) {
    std::vector<int>       int_vec{1, 2, 3};
    std::vector<int> const int_vec_const{1, 2, 3, 4};

    constexpr ParameterType                                   ptype = ParameterType::send_counts;
    constexpr BufferType                                      btype = BufferType::in_buffer;
    ContainerBasedConstBuffer<std::vector<int>, ptype, btype> buffer_based_on_int_vector(int_vec);
    ContainerBasedConstBuffer<std::vector<int>, ptype, btype> buffer_based_on_const_int_vector(int_vec_const);

    // the following just has to compile
    {
        [[maybe_unused]] auto buffer1 = buffer_based_on_int_vector;
        [[maybe_unused]] auto buffer2 = buffer_based_on_const_int_vector;
    }

    {
        [[maybe_unused]] auto buffer1 = std::move(buffer_based_on_int_vector);
        [[maybe_unused]] auto buffer2 = std::move(buffer_based_on_const_int_vector);
    }
}
