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

#include "gtest/gtest.h"
#include <array>
#include <deque>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/assertion_levels.hpp"
#include "kamping/parameter_objects.hpp"
#include "legacy_parameter_objects.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

TEST(IsSpecializationTest, is_specialization_basics) {
    EXPECT_TRUE((kamping::internal::is_specialization<std::vector<int>, std::vector>::value));
    EXPECT_TRUE((kamping::internal::is_specialization<std::vector<bool>, std::vector>::value));
    EXPECT_TRUE(
        (kamping::internal::is_specialization<std::vector<int, testing::CustomAllocator<int>>, std::vector>::value));
    EXPECT_TRUE((kamping::internal::is_specialization<
                 std::vector<double, testing::CustomAllocator<double>>, std::vector>::value));
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
    EXPECT_TRUE((kamping::internal::is_vector_bool_v<std::vector<bool, testing::CustomAllocator<bool>>>));
    EXPECT_TRUE(kamping::internal::is_vector_bool_v<const std::vector<bool>>);
    EXPECT_TRUE(kamping::internal::is_vector_bool_v<std::vector<bool>&>);
    EXPECT_TRUE(kamping::internal::is_vector_bool_v<std::vector<bool> const&>);
    EXPECT_FALSE(kamping::internal::is_vector_bool_v<std::vector<int>>);
    EXPECT_FALSE((kamping::internal::is_vector_bool_v<std::vector<int, testing::CustomAllocator<int>>>));
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
    constexpr ParameterType              ptype = ParameterType::send_counts;
    EmptyBuffer<std::vector<int>, ptype> empty_buffer{};

    EXPECT_EQ(empty_buffer.size(), 0);
    EXPECT_EQ(empty_buffer.get().size(), 0);
    EXPECT_EQ(empty_buffer.get().data(), nullptr);
    EXPECT_EQ(empty_buffer.data(), nullptr);
}

// Tests the basic functionality of ContainerBasedConstBuffer
TEST(ContainerBasedConstBufferTest, get_basics) {
    std::vector<int>       int_vec{1, 2, 3};
    std::vector<int> const int_vec_const{1, 2, 3, 4};

    constexpr ParameterType                            ptype = ParameterType::send_counts;
    ContainerBasedConstBuffer<std::vector<int>, ptype> buffer_based_on_int_vector(int_vec);
    ContainerBasedConstBuffer<std::vector<int>, ptype> buffer_based_on_const_int_vector(int_vec_const);

    EXPECT_EQ(buffer_based_on_int_vector.size(), int_vec.size());
    EXPECT_EQ(buffer_based_on_int_vector.get().size(), int_vec.size());
    EXPECT_EQ(buffer_based_on_int_vector.get().data(), int_vec.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_int_vector.get().data()), const int*>);
    EXPECT_EQ(buffer_based_on_int_vector.data(), int_vec.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_int_vector.data()), const int*>);

    EXPECT_EQ(buffer_based_on_const_int_vector.get().size(), int_vec_const.size());
    EXPECT_EQ(buffer_based_on_const_int_vector.get().data(), int_vec_const.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_const_int_vector.get().data()), const int*>);
    EXPECT_EQ(buffer_based_on_const_int_vector.data(), int_vec_const.data());
    static_assert(std::is_same_v<decltype(buffer_based_on_const_int_vector.data()), const int*>);
}

TEST(ContainerBasedConstBufferTest, get_containers_other_than_vector) {
    std::string                                                  str = "I am underlying storage";
    testing::OwnContainer<int>                                   own_container;
    constexpr ParameterType                                      ptype = ParameterType::send_buf;
    ContainerBasedConstBuffer<std::string, ptype>                buffer_based_on_string(str);
    ContainerBasedConstBuffer<testing::OwnContainer<int>, ptype> buffer_based_on_own_container(own_container);

    EXPECT_EQ(buffer_based_on_string.get().size(), str.size());
    EXPECT_EQ(buffer_based_on_string.get().data(), str.data());

    EXPECT_EQ(buffer_based_on_own_container.get().size(), own_container.size());
    EXPECT_EQ(buffer_based_on_own_container.get().data(), own_container.data());
}

TEST(ContainerBasedConstBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                            ptype = ParameterType::send_counts;
    const std::vector<int>                             container{1, 2, 3};
    ContainerBasedConstBuffer<std::vector<int>, ptype> buffer1(container);
    ContainerBasedConstBuffer<std::vector<int>, ptype> buffer2(std::move(buffer1));
    EXPECT_EQ(buffer2.get().size(), container.size());
    EXPECT_TRUE(std::equal(container.begin(), container.end(), buffer2.get().data()));
}

// Tests the basic functionality of ContainerBasedOwningBuffer
TEST(ContainerBasedOwningBufferTest, get_basics) {
    std::vector<int> int_vec{1, 2, 3};

    constexpr ParameterType                             ptype = ParameterType::send_counts;
    ContainerBasedOwningBuffer<std::vector<int>, ptype> buffer_based_on_moved_vector(std::move(int_vec));
    ContainerBasedOwningBuffer<std::vector<int>, ptype> buffer_based_on_rvalue_vector(std::vector<int>{1, 2, 3});

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

    EXPECT_EQ(buffer_based_on_rvalue_vector.size(), 3);
    EXPECT_EQ(buffer_based_on_rvalue_vector.get().size(), 3);
    EXPECT_EQ(buffer_based_on_rvalue_vector.get().data()[0], 1);
    EXPECT_EQ(buffer_based_on_rvalue_vector.get().data()[1], 2);
    EXPECT_EQ(buffer_based_on_rvalue_vector.get().data()[2], 3);
    static_assert(std::is_same_v<decltype(buffer_based_on_rvalue_vector.get().data()), const int*>);
    EXPECT_EQ(buffer_based_on_rvalue_vector.data()[0], 1);
    EXPECT_EQ(buffer_based_on_rvalue_vector.data()[1], 2);
    EXPECT_EQ(buffer_based_on_rvalue_vector.data()[2], 3);
    static_assert(std::is_same_v<decltype(buffer_based_on_rvalue_vector.data()), int const*>);

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

    // string
    std::string                                    str      = "I am underlying storage";
    std::string                                    expected = "I am underlying storage";
    ContainerBasedOwningBuffer<std::string, ptype> buffer_based_on_string(std::move(str));

    EXPECT_EQ(buffer_based_on_string.get().size(), expected.size());
    EXPECT_EQ(
        std::string(
            buffer_based_on_string.get().data(),
            buffer_based_on_string.get().data() + buffer_based_on_string.get().size()),
        expected);
    {
        auto const& underlying_container = buffer_based_on_string.underlying();
        EXPECT_EQ(underlying_container, expected);
    }
    // own container
    testing::OwnContainer<int> own_container{1, 2, 3};
    EXPECT_EQ(own_container.copy_count(), 0);

    ContainerBasedOwningBuffer<testing::OwnContainer<int>, ptype> buffer_based_on_own_container(
        std::move(own_container));
    EXPECT_EQ(own_container.copy_count(), 0);
    EXPECT_EQ(buffer_based_on_own_container.underlying().copy_count(), 0);

    EXPECT_EQ(buffer_based_on_own_container.get().size(), 3);
    EXPECT_EQ(buffer_based_on_own_container.get().data()[0], 1);
    EXPECT_EQ(buffer_based_on_own_container.get().data()[1], 2);
    EXPECT_EQ(buffer_based_on_own_container.get().data()[2], 3);
    {
        auto const& underlying_container = buffer_based_on_own_container.underlying();
        EXPECT_EQ(underlying_container, (testing::OwnContainer<int>{1, 2, 3}));
    }
}

TEST(ContainerBasedOwningBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                             ptype = ParameterType::send_counts;
    const std::vector<int>                              container{1, 2, 3};
    ContainerBasedOwningBuffer<std::vector<int>, ptype> buffer1({1, 2, 3});
    ContainerBasedOwningBuffer<std::vector<int>, ptype> buffer2(std::move(buffer1));
    EXPECT_EQ(buffer2.get().size(), 3);

    const std::vector<int> expected_container{1, 2, 3};
    EXPECT_TRUE(std::equal(expected_container.begin(), expected_container.end(), buffer2.get().data()));
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

TEST(UserAllocatedContainerBasedBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType ptype = ParameterType::send_counts;
    std::vector<int>        container{1, 2, 3};
    const auto              const_container = container; // ensure that container is not altered
    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype> buffer1(container);
    UserAllocatedContainerBasedBuffer<std::vector<int>, ptype> buffer2(std::move(buffer1));
    EXPECT_EQ(buffer2.get().size(), const_container.size());
    EXPECT_TRUE(std::equal(const_container.begin(), const_container.end(), buffer2.get().data()));
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

TEST(LibAllocatedContainerBasedBufferTest, move_ctor_assignment_operator_is_enabled) {
    constexpr ParameterType                                             ptype = ParameterType::recv_counts;
    LibAllocatedContainerBasedBuffer<testing::OwnContainer<int>, ptype> buffer1;
    const size_t                                                        size = 3;
    buffer1.resize(size);
    buffer1.get().data()[0] = 0;
    buffer1.get().data()[1] = 1;
    buffer1.get().data()[2] = 2;
    LibAllocatedContainerBasedBuffer<testing::OwnContainer<int>, ptype> buffer2(std::move(buffer1));
    LibAllocatedContainerBasedBuffer<testing::OwnContainer<int>, ptype> buffer3;
    buffer3 = std::move(buffer2);
    EXPECT_EQ(buffer3.get().size(), 3);
    EXPECT_EQ(buffer3.get().data()[0], 0);
    EXPECT_EQ(buffer3.get().data()[1], 1);
    EXPECT_EQ(buffer3.get().data()[2], 2);
}

TEST(SingleElementConstBufferTest, get_basics) {
    constexpr ParameterType              ptype = ParameterType::send_counts;
    int                                  value = 5;
    SingleElementConstBuffer<int, ptype> int_buffer(value);

    EXPECT_EQ(int_buffer.size(), 1);
    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);
    EXPECT_EQ(*(int_buffer.data()), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_FALSE(int_buffer.is_modifiable);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, decltype(value)>);
}

TEST(SingleElementConstBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType              ptype = ParameterType::send_counts;
    const int                            elem  = 42;
    SingleElementConstBuffer<int, ptype> buffer1(elem);
    SingleElementConstBuffer<int, ptype> buffer2(std::move(buffer1));
    EXPECT_EQ(*buffer2.get().data(), elem);
    EXPECT_EQ(*buffer2.data(), elem);
    EXPECT_EQ(buffer2.get_single_element(), elem);
}

TEST(SingleElementOwningBufferTest, get_basics) {
    constexpr ParameterType               ptype = ParameterType::send_counts;
    SingleElementOwningBuffer<int, ptype> int_buffer(5);

    EXPECT_EQ(int_buffer.size(), 1);
    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);
    EXPECT_EQ(*(int_buffer.data()), 5);
    EXPECT_EQ(int_buffer.underlying(), 5);
    EXPECT_EQ(int_buffer.get_single_element(), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_FALSE(int_buffer.is_modifiable);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, int>);
}

TEST(SingleElementOwningBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType               ptype = ParameterType::send_counts;
    SingleElementOwningBuffer<int, ptype> buffer1(42);
    SingleElementOwningBuffer<int, ptype> buffer2(std::move(buffer1));
    EXPECT_EQ(*buffer2.get().data(), 42);
    EXPECT_EQ(*buffer2.data(), 42);
    EXPECT_EQ(buffer2.underlying(), 42);
    EXPECT_EQ(buffer2.get_single_element(), 42);
}

TEST(SingleElementModifiableBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                   ptype      = ParameterType::send_counts;
    int                                       elem       = 42;
    const int                                 const_elem = elem;
    SingleElementModifiableBuffer<int, ptype> buffer1(elem);
    SingleElementModifiableBuffer<int, ptype> buffer2(std::move(buffer1));
    EXPECT_EQ(*buffer2.get().data(), const_elem);
    EXPECT_EQ(*buffer2.data(), const_elem);
    EXPECT_EQ(buffer2.get_single_element(), const_elem);
}

TEST(SingleElementModifiableBufferTest, get_basics) {
    constexpr ParameterType                   ptype = ParameterType::send_counts;
    int                                       value = 5;
    SingleElementModifiableBuffer<int, ptype> int_buffer(value);

    EXPECT_EQ(int_buffer.size(), 1);
    int_buffer.resize(1);
    EXPECT_EQ(int_buffer.size(), 1);
#if KASSERT_ASSERTION_LEVEL >= KAMPING_ASSERTION_LEVEL_NORMAL
    EXPECT_DEATH(
        int_buffer.resize(0), "Cannot resize a single element buffer to hold zero or more than one element. Single "
                              "element buffers always hold exactly one element.");
    EXPECT_DEATH(
        int_buffer.resize(2), "Cannot resize a single element buffer to hold zero or more than one element. Single "
                              "element buffers always hold exactly one element.");
#endif

    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);
    EXPECT_EQ(*(int_buffer.data()), 5);
    EXPECT_EQ(int_buffer.get_single_element(), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_TRUE(int_buffer.is_modifiable);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, decltype(value)>);
}

TEST(LibAllocatedSingleElementBufferTest, move_constructor_is_enabled) {
    constexpr ParameterType                     ptype      = ParameterType::send_counts;
    int                                         elem       = 42;
    const int                                   const_elem = elem;
    LibAllocatedSingleElementBuffer<int, ptype> buffer1{};
    *buffer1.get().data() = elem;
    LibAllocatedSingleElementBuffer<int, ptype> buffer2(std::move(buffer1));
    EXPECT_EQ(*buffer2.get().data(), const_elem);
    EXPECT_EQ(*buffer2.data(), const_elem);
    EXPECT_EQ(buffer2.get_single_element(), const_elem);
}

TEST(LibAllocatedSingleElementBufferTest, get_basics) {
    constexpr ParameterType                     ptype = ParameterType::send_counts;
    int                                         value = 5;
    LibAllocatedSingleElementBuffer<int, ptype> int_buffer{};

    *int_buffer.get().data() = value;

    EXPECT_EQ(int_buffer.size(), 1);
    int_buffer.resize(1);
    EXPECT_EQ(int_buffer.size(), 1);
#if KASSERT_ASSERTION_LEVEL >= KAMPING_ASSERTION_LEVEL_NORMAL
    EXPECT_DEATH(
        int_buffer.resize(0), "Cannot resize a single element buffer to hold zero or more than one element. Single "
                              "element buffers always hold exactly one element.");
    EXPECT_DEATH(
        int_buffer.resize(2), "Cannot resize a single element buffer to hold zero or more than one element. Single "
                              "element buffers always hold exactly one element.");
#endif
    EXPECT_EQ(int_buffer.get().size(), 1);
    EXPECT_EQ(*(int_buffer.get().data()), 5);
    EXPECT_EQ(*(int_buffer.data()), 5);
    EXPECT_EQ(int_buffer.get_single_element(), 5);

    EXPECT_EQ(decltype(int_buffer)::parameter_type, ptype);
    EXPECT_TRUE(int_buffer.is_modifiable);

    static_assert(std::is_same_v<decltype(int_buffer)::value_type, decltype(value)>);

    int extracted_value = int_buffer.extract();
    EXPECT_EQ(extracted_value, value);
}

TEST(RootTest, move_constructor_assignment_operator_is_enabled) {
    int       rank       = 2;
    const int const_rank = rank;
    Root      root1(rank);
    Root      root2 = std::move(root1);
    Root      root3(rank + 1);
    root3 = std::move(root2);
    EXPECT_EQ(root3.rank(), const_rank);
}

TEST(OperationBuilderTest, move_constructor_assignment_operator_is_enabled) {
    // simply test that move ctor and assignment operator can be called.
    OperationBuilder op_builder1(ops::plus<>(), commutative);
    OperationBuilder op_builder2(std::move(op_builder1));
    OperationBuilder op_builder3(ops::plus<>(), commutative);
    op_builder3 = std::move(op_builder2);
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

TEST(DataBufferTest, has_extract) {
    static_assert(
        has_extract_v<DataBuffer<
            int, ParameterType::send_buf, BufferModifiability::modifiable, BufferOwnership::owning,
            BufferAllocation::lib_allocated>>,
        "Library allocated DataBuffers must have an extract() member function");
    static_assert(
        !has_extract_v<DataBuffer<
            int, ParameterType::send_buf, BufferModifiability::modifiable, BufferOwnership::owning,
            BufferAllocation::user_allocated>>,
        "User allocated DataBuffers must not have an extract() member function");
}

TEST(ParameterFactoriesTest, is_int_type) {
    EXPECT_FALSE(is_int_type(kamping::internal::ParameterType::send_buf));
    EXPECT_FALSE(is_int_type(kamping::internal::ParameterType::recv_buf));
    EXPECT_FALSE(is_int_type(kamping::internal::ParameterType::send_recv_buf));
    EXPECT_TRUE(is_int_type(kamping::internal::ParameterType::recv_counts));
    EXPECT_TRUE(is_int_type(kamping::internal::ParameterType::recv_displs));
    EXPECT_TRUE(is_int_type(kamping::internal::ParameterType::recv_count));
    EXPECT_TRUE(is_int_type(kamping::internal::ParameterType::send_counts));
    EXPECT_TRUE(is_int_type(kamping::internal::ParameterType::send_displs));
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST(LibAllocatedContainerBasedBufferTest, prevent_usage_after_extraction) {
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_buf> buffer;

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
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_buf>    recv_buffer;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_counts> recv_counts;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_count>  recv_count;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::recv_displs> recv_displs;
    LibAllocatedContainerBasedBuffer<std::vector<int>, ParameterType::send_displs> send_displs;

    MPIResult result(
        std::move(recv_buffer), std::move(recv_counts), std::move(recv_count), std::move(recv_displs),
        std::move(send_displs));

    std::ignore = result.extract_recv_buffer();
    EXPECT_KASSERT_FAILS(result.extract_recv_buffer(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_recv_counts();
    EXPECT_KASSERT_FAILS(result.extract_recv_counts(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_recv_displs();
    EXPECT_KASSERT_FAILS(result.extract_recv_displs(), "Cannot extract a buffer that has already been extracted.");

    std::ignore = result.extract_send_displs();
    EXPECT_KASSERT_FAILS(result.extract_send_displs(), "Cannot extract a buffer that has already been extracted.");
}
#endif
