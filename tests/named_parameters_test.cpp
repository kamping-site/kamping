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

#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "helpers_for_testing.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/parameter_objects.hpp"
#include "legacy_parameter_objects.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

namespace testing {
template <typename ExpectedValueType, typename GeneratedBuffer, typename T>
void test_const_referencing_buffer(
    GeneratedBuffer&                 generated_buffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    Span<T>&                         expected_span
) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_FALSE(GeneratedBuffer::is_modifiable);
    EXPECT_FALSE(GeneratedBuffer::is_owning);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    static_assert(
        std::is_const_v<std::remove_pointer_t<decltype(generated_buffer.data())>>,
        "Member data() of the generated buffer does not point to const memory."
    );
    static_assert(
        std::is_const_v<std::remove_reference_t<decltype(generated_buffer.underlying())>>,
        "Member underlying() of the generated buffer provides access to non-const memory."
    );

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
void test_const_owning_buffer(
    GeneratedBuffer&                 generated_buffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    ExpectedValueContainer&&         expected_value_container
) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_FALSE(GeneratedBuffer::is_modifiable);
    EXPECT_TRUE(GeneratedBuffer::is_owning);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    static_assert(
        std::is_const_v<std::remove_pointer_t<decltype(generated_buffer.data())>>,
        "Member data() of the generated buffer does not point to const memory."
    );
    static_assert(
        std::is_const_v<std::remove_reference_t<decltype(generated_buffer.underlying())>>,
        "Member underlying() of the generated buffer provides access to non-const memory."
    );

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
    Span<T>&                         expected_span
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
    kamping::BufferResizePolicy      expected_resize_policy,
    UnderlyingContainer&             underlying_container
) {
    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_TRUE(GeneratedBuffer::is_modifiable);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);
    EXPECT_EQ(GeneratedBuffer::resize_policy, expected_resize_policy);

    auto resize_write_check = [&](size_t nb_elements) {
        if constexpr (GeneratedBuffer::resize_policy != BufferResizePolicy::no_resize) {
            generated_buffer.resize(nb_elements);
        }
        if (nb_elements <= generated_buffer.size()) {
            ExpectedValueType* ptr = generated_buffer.data();
            EXPECT_EQ(ptr, std::data(underlying_container));
            for (size_t i = 0; i < nb_elements; ++i) {
                ptr[i] = static_cast<ExpectedValueType>(nb_elements - i);
                EXPECT_EQ(ptr[i], underlying_container[i]);
            }
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
    EXPECT_EQ(GeneratedBuffer::resize_policy, BufferResizePolicy::resize_to_fit);

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

TEST(ParamterFactoriesTest, test_type_list) {
    using my_type_list = internal::type_list<int, double, std::string>;
    ASSERT_TRUE(my_type_list::contains<int>);
    ASSERT_TRUE(my_type_list::contains<double>);
    ASSERT_TRUE(my_type_list::contains<std::string>);
    ASSERT_FALSE(my_type_list::contains<char>);
    ASSERT_FALSE(my_type_list::contains<float>);
}

TEST(ParameterFactoriesTest, send_buf_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_buf(int_vec).construct_buffer_or_rebind();
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_buf_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_buf(const_int_vec).construct_buffer_or_rebind();
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_buf_basics_moved_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    std::vector<int> const expected          = const_int_vec;
    auto                   gen_via_moved_vec = send_buf(std::move(const_int_vec)).construct_buffer_or_rebind();
    using ExpectedValueType                  = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_moved_vec,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_buf_basics_vector_from_function) {
    auto make_vector = []() {
        std::vector<int> vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
        return vec;
    };
    std::vector<int> const expected                  = make_vector();
    auto                   gen_via_vec_from_function = send_buf(make_vector()).construct_buffer_or_rebind();
    using ExpectedValueType                          = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_vec_from_function,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_buf_basics_vector_from_initializer_list) {
    std::vector<int> expected      = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto gen_via_vec_from_function = send_buf({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1}).construct_buffer_or_rebind();
    using ExpectedValueType        = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_vec_from_function,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_buf_single_element) {
    {
        uint8_t value                     = 11;
        auto    gen_single_element_buffer = send_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            value
        );
    }
    {
        uint16_t value                     = 4211;
        auto     gen_single_element_buffer = send_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            value
        );
    }
    {
        uint32_t value                     = 4096;
        auto     gen_single_element_buffer = send_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            value
        );
    }
    {
        uint64_t value                     = 555555;
        auto     gen_single_element_buffer = send_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            value
        );
    }
    {
        // pass value as rvalue
        auto gen_single_element_buffer = send_buf(42051).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            42051
        );
    }
    {
        struct CustomType {
            uint64_t v1;
            int      v2;
            char     v3;

            bool operator==(CustomType const& other) const {
                return std::tie(v1, v2, v3) == std::tie(other.v1, other.v2, other.v3);
            }
        }; // struct CustomType
        {
            CustomType value                     = {843290834, -482, 'a'};
            auto       gen_single_element_buffer = send_buf(value).construct_buffer_or_rebind();
            testing::test_single_element_buffer(
                gen_single_element_buffer,
                ParameterType::send_buf,
                internal::BufferType::in_buffer,
                value
            );
        }
        {
            auto gen_single_element_buffer = send_buf(CustomType{843290834, -482, 'a'}).construct_buffer_or_rebind();
            testing::test_single_element_buffer(
                gen_single_element_buffer,
                ParameterType::send_buf,
                internal::BufferType::in_buffer,
                CustomType{843290834, -482, 'a'}
            );
        }
    }
}

TEST(ParameterFactoriesTest, send_buf_switch) {
    uint8_t              value  = 0;
    std::vector<uint8_t> values = {0, 0, 0, 0, 0, 0};

    [[maybe_unused]] auto gen_single_element_buffer        = send_buf(value).construct_buffer_or_rebind();
    [[maybe_unused]] auto gen_int_vec_buffer               = send_buf(values).construct_buffer_or_rebind();
    [[maybe_unused]] auto gen_single_element_owning_buffer = send_buf(uint8_t(0)).construct_buffer_or_rebind();
    [[maybe_unused]] auto gen_int_vec_owning_buffer =
        send_buf(std::vector<uint8_t>{0, 0, 0, 0, 0, 0}).construct_buffer_or_rebind();

    bool const single_result = std::is_same_v<
        decltype(gen_single_element_buffer),
        SingleElementConstBuffer<uint8_t, ParameterType::send_buf, BufferType::in_buffer>>;
    EXPECT_TRUE(single_result);
    bool const vec_result = std::is_same_v<
        decltype(gen_int_vec_buffer),
        ContainerBasedConstBuffer<std::vector<uint8_t>, ParameterType::send_buf, BufferType::in_buffer>>;
    EXPECT_TRUE(vec_result);
    bool const owning_single_result = std::is_same_v<
        decltype(gen_single_element_owning_buffer),
        SingleElementOwningBuffer<uint8_t, ParameterType::send_buf, BufferType::in_buffer>>;
    EXPECT_TRUE(owning_single_result);
    bool const owning_vec_result = std::is_same_v<
        decltype(gen_int_vec_owning_buffer),
        ContainerBasedOwningBuffer<std::vector<uint8_t>, ParameterType::send_buf, BufferType::in_buffer>>;
    EXPECT_TRUE(owning_vec_result);
}

TEST(ParameterFactoriesTest, send_buf_ignored) {
    auto ignored_send_buf = send_buf(ignore<int>).construct_buffer_or_rebind();
    EXPECT_EQ(ignored_send_buf.get().data(), nullptr);
    EXPECT_EQ(ignored_send_buf.get().size(), 0);
}

TEST(ParameterFactoriesTest, send_buf_owning_move_only_data) {
    // test that data within the buffer is still treated as constant but can be returned without being copied
    testing::NonCopyableOwnContainer<int>       vec{1, 2, 3, 4}; // required as original data will be moved to buffer
    testing::NonCopyableOwnContainer<int> const expected_vec{
        1,
        2,
        3,
        4}; // required as original data will be moved to buffer
    auto send_buffer = send_buf(std::move(vec)).construct_buffer_or_rebind();
    testing::test_const_owning_buffer<int>(
        send_buffer,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected_vec
    );
    auto extracted_vec = send_buffer.extract();
    EXPECT_THAT(extracted_vec, testing::ElementsAre(1, 2, 3, 4));
}

TEST(ParameterFactoriesTest, send_counts_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_counts(int_vec).construct_buffer_or_rebind();
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_counts_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_counts(const_int_vec).construct_buffer_or_rebind();
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_counts_basics_moved_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             expected        = int_vec;
    auto             gen_via_int_vec = send_counts(std::move(int_vec)).construct_buffer_or_rebind();
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_counts_basics_initializer_list) {
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto gen_via_int_initializer_list = send_counts({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1}).construct_buffer_or_rebind();
    using ExpectedValueType           = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_int_initializer_list,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_counts_owning_move_only_data) {
    // test that data within the buffer is still treated as constant but can be returned without being copied
    testing::NonCopyableOwnContainer<int>       vec{1, 2, 3, 4}; // required as original data will be moved to buffer
    testing::NonCopyableOwnContainer<int> const expected_vec{
        1,
        2,
        3,
        4}; // required as original data will be moved to buffer
    auto send_buffer = send_counts(std::move(vec)).construct_buffer_or_rebind();
    testing::test_const_owning_buffer<int>(
        send_buffer,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected_vec
    );
    auto extracted_vec = send_buffer.extract();
    EXPECT_THAT(extracted_vec, testing::ElementsAre(1, 2, 3, 4));
}

TEST(ParameterFactoriesTest, recv_counts_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = recv_counts(int_vec).construct_buffer_or_rebind();
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::recv_counts,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, recv_counts_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = recv_counts(const_int_vec).construct_buffer_or_rebind();
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::recv_counts,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, recv_counts_in_basics_moved_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             expected          = int_vec;
    auto             gen_via_moved_vec = recv_counts(std::move(int_vec)).construct_buffer_or_rebind();
    using ExpectedValueType            = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_moved_vec,
        ParameterType::recv_counts,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, recv_counts_in_basics_initializer_list) {
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto gen_via_initializer_list = recv_counts({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1}).construct_buffer_or_rebind();
    using ExpectedValueType       = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_initializer_list,
        ParameterType::recv_counts,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_displs_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_displs(int_vec).construct_buffer_or_rebind();
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_displs_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_displs(const_int_vec).construct_buffer_or_rebind();
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_displs_in_basics_moved_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             expected          = int_vec;
    auto             gen_via_moved_vec = send_displs(std::move(int_vec)).construct_buffer_or_rebind();
    using ExpectedValueType            = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_moved_vec,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_displs_in_basics_initializer_list) {
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto gen_via_initializer_list = send_displs({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1}).construct_buffer_or_rebind();
    using ExpectedValueType       = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_initializer_list,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_displs_owning_move_only_data) {
    // test that data within the buffer is still treated as constant but can be returned without being copied
    testing::NonCopyableOwnContainer<int>       vec{1, 2, 3, 4}; // required as original data will be moved to buffer
    testing::NonCopyableOwnContainer<int> const expected_vec{
        1,
        2,
        3,
        4}; // required as original data will be moved to buffer
    auto send_buffer = send_displs(std::move(vec)).construct_buffer_or_rebind();
    testing::test_const_owning_buffer<int>(
        send_buffer,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected_vec
    );
    auto extracted_vec = send_buffer.extract();
    EXPECT_THAT(extracted_vec, testing::ElementsAre(1, 2, 3, 4));
}

TEST(ParameterFactoriesTest, recv_displs_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = recv_displs(int_vec).construct_buffer_or_rebind();
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::recv_displs,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, recv_displs_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = recv_displs(const_int_vec).construct_buffer_or_rebind();
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::recv_displs,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, recv_displs_in_basics_moved_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             expected          = int_vec;
    auto             gen_via_moved_vec = recv_displs(std::move(int_vec)).construct_buffer_or_rebind();
    using ExpectedValueType            = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_moved_vec,
        ParameterType::recv_displs,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, recv_displs_in_basics_initializer_list) {
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto gen_via_initializer_list = recv_displs({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1}).construct_buffer_or_rebind();
    using ExpectedValueType       = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_initializer_list,
        ParameterType::recv_displs,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, recv_buf_basics_user_alloc) {
    size_t const     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_on_user_alloc_vector = recv_buf(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                      = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector,
        ParameterType::recv_buf,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_recv_buf_basics_user_alloc) {
    std::vector<int> int_vec;
    auto             buffer_on_user_alloc_vector =
        recv_buf<BufferResizePolicy::resize_to_fit>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector,
        ParameterType::recv_buf,
        internal::BufferType::out_buffer,
        BufferResizePolicy::resize_to_fit,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_if_required_recv_buf_basics_user_alloc) {
    std::vector<int> int_vec;
    auto buffer_on_user_alloc_vector = recv_buf<BufferResizePolicy::grow_only>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType          = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector,
        ParameterType::recv_buf,
        internal::BufferType::out_buffer,
        BufferResizePolicy::grow_only,
        int_vec
    );
}

TEST(ParameterFactoriesTest, recv_buf_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_buf,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_buf_basics_library_alloc_container_of) {
    auto buffer_based_on_library_alloc_vector =
        recv_buf(alloc_container_of<int>).template construct_buffer_or_rebind<std::vector>();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_buf,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_buf_basics_library_alloc_container_of_with_own_container) {
    auto buffer_based_on_library_alloc_vector =
        recv_buf(alloc_container_of<int>).template construct_buffer_or_rebind<testing::OwnContainer>();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_buf,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, send_counts_out_basics_user_alloc) {
    size_t const     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_vector = send_counts_out(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                            = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::send_counts,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int_vec
    );
}

TEST(ParameterFactoriesTest, always_resizing_send_counts_out_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::resize_to_fit;
    auto buffer_based_on_user_alloc_vector = send_counts_out<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::send_counts,
        internal::BufferType::out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_if_required_send_counts_out_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::grow_only;
    auto buffer_based_on_user_alloc_vector = send_counts_out<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::send_counts,
        internal::BufferType::out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, send_counts_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector =
        send_counts_out(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_counts,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, send_counts_out_basics_library_alloc_without_explicit_type) {
    auto buffer_based_on_library_alloc_vector =
        send_counts_out(alloc_new_using<std::vector>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_counts,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, send_displs_out_basics_user_alloc) {
    size_t const     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_vector = send_displs_out(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                            = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::send_displs,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_send_displs_out_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::resize_to_fit;
    auto buffer_based_on_user_alloc_vector = send_displs_out<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::send_displs,
        internal::BufferType::out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_if_required_send_displs_out_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::grow_only;
    auto buffer_based_on_user_alloc_vector = send_displs_out<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::send_displs,
        internal::BufferType::out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, send_displs_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector =
        send_displs_out(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_displs,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, send_displs_out_basics_library_alloc_without_explicit_type) {
    auto buffer_based_on_library_alloc_vector =
        send_displs_out(alloc_new_using<std::vector>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_displs,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_counts_out_basics_user_alloc) {
    size_t const     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_buffer = recv_counts_out(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                            = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_buffer,
        ParameterType::recv_counts,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_recv_counts_out_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::resize_to_fit;
    auto buffer_based_on_user_alloc_buffer = recv_counts_out<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_buffer,
        ParameterType::recv_counts,
        internal::BufferType::out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_if_required_recv_counts_out_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::grow_only;
    auto buffer_based_on_user_alloc_buffer = recv_counts_out<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_buffer,
        ParameterType::recv_counts,
        internal::BufferType::out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, recv_counts_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector =
        recv_counts_out(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_counts,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_counts_out_basics_library_alloc_without_explicit_type) {
    auto buffer_based_on_library_alloc_vector =
        recv_counts_out(alloc_new_using<std::vector>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_counts,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_displs_out_basics_user_alloc) {
    size_t const     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_vector = recv_displs_out(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                            = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::recv_displs,
        internal::BufferType::out_buffer,
        BufferResizePolicy::no_resize,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_recv_displs_out_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::resize_to_fit;
    auto buffer_based_on_user_alloc_vector = recv_displs_out<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::recv_displs,
        internal::BufferType::out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_if_required_recv_displs_out_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::grow_only;
    auto buffer_based_on_user_alloc_vector = recv_displs_out<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::recv_displs,
        internal::BufferType::out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, recv_displs_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector =
        recv_displs_out(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_displs,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_displs_out_basics_library_alloc_without_explicit_type) {
    auto buffer_based_on_library_alloc_vector =
        recv_displs_out(alloc_new_using<std::vector>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_displs,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, send_count_in) {
    auto param       = send_count(42).construct_buffer_or_rebind();
    using param_type = std::remove_reference_t<decltype(param)>;
    EXPECT_EQ(param.size(), 1);
    EXPECT_EQ(param.underlying(), 42);
    EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
    EXPECT_EQ(param_type::parameter_type, ParameterType::send_count);
    EXPECT_EQ(param_type::buffer_type, BufferType::in_buffer);
    EXPECT_FALSE(param_type::is_modifiable);
}

TEST(ParameterFactoriesTest, send_count_out) {
    { // lib-allocated memory
        auto param       = send_count_out().construct_buffer_or_rebind();
        using param_type = std::remove_reference_t<decltype(param)>;
        EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
        EXPECT_EQ(param_type::parameter_type, ParameterType::send_count);
        EXPECT_EQ(param_type::buffer_type, BufferType::out_buffer);
        EXPECT_TRUE(param_type::is_modifiable);
        param.underlying() = 42;
        EXPECT_EQ(param.get_single_element(), 42);
        EXPECT_EQ(param.extract(), 42);
    }
    { // user-allocated memory
        int  count       = -1;
        auto param       = send_count_out(count).construct_buffer_or_rebind();
        using param_type = std::remove_reference_t<decltype(param)>;
        EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
        EXPECT_EQ(param_type::parameter_type, ParameterType::send_count);
        EXPECT_EQ(param_type::buffer_type, BufferType::out_buffer);
        EXPECT_TRUE(param_type::is_modifiable);
        EXPECT_EQ(param.get_single_element(), -1);
        param.underlying() = 42;
        EXPECT_EQ(param.get_single_element(), 42);
        EXPECT_EQ(count, 42);
    }
}

TEST(ParameterFactoriesTest, recv_count_in) {
    auto param       = recv_count(42).construct_buffer_or_rebind();
    using param_type = std::remove_reference_t<decltype(param)>;
    EXPECT_EQ(param.size(), 1);
    EXPECT_EQ(param.underlying(), 42);
    EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
    EXPECT_EQ(param_type::parameter_type, ParameterType::recv_count);
    EXPECT_EQ(param_type::buffer_type, BufferType::in_buffer);
    EXPECT_FALSE(param_type::is_modifiable);
}

TEST(ParameterFactoriesTest, recv_count_out) {
    { // lib-allocated memory
        auto param       = recv_count_out().construct_buffer_or_rebind();
        using param_type = std::remove_reference_t<decltype(param)>;
        EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
        EXPECT_EQ(param_type::parameter_type, ParameterType::recv_count);
        EXPECT_EQ(param_type::buffer_type, BufferType::out_buffer);
        EXPECT_TRUE(param_type::is_modifiable);
        param.underlying() = 42;
        EXPECT_EQ(param.get_single_element(), 42);
        EXPECT_EQ(param.extract(), 42);
    }
    { // user-allocated memory
        int  count       = -1;
        auto param       = recv_count_out(count).construct_buffer_or_rebind();
        using param_type = std::remove_reference_t<decltype(param)>;
        EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
        EXPECT_EQ(param_type::parameter_type, ParameterType::recv_count);
        EXPECT_EQ(param_type::buffer_type, BufferType::out_buffer);
        EXPECT_TRUE(param_type::is_modifiable);
        EXPECT_EQ(param.get_single_element(), -1);
        param.underlying() = 42;
        EXPECT_EQ(param.get_single_element(), 42);
        EXPECT_EQ(count, 42);
    }
}

TEST(ParameterFactoriesTest, send_recv_count_in) {
    auto param       = send_recv_count(42).construct_buffer_or_rebind();
    using param_type = std::remove_reference_t<decltype(param)>;
    EXPECT_EQ(param.size(), 1);
    EXPECT_EQ(param.underlying(), 42);
    EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
    EXPECT_EQ(param_type::parameter_type, ParameterType::send_recv_count);
    EXPECT_EQ(param_type::buffer_type, BufferType::in_buffer);
    EXPECT_FALSE(param_type::is_modifiable);
}

TEST(ParameterFactoriesTest, send_recv_count_out) {
    { // lib-allocated memory
        auto param       = send_recv_count_out().construct_buffer_or_rebind();
        using param_type = std::remove_reference_t<decltype(param)>;
        EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
        EXPECT_EQ(param_type::parameter_type, ParameterType::send_recv_count);
        EXPECT_EQ(param_type::buffer_type, BufferType::out_buffer);
        EXPECT_TRUE(param_type::is_modifiable);
        param.underlying() = 42;
        EXPECT_EQ(param.get_single_element(), 42);
        EXPECT_EQ(param.extract(), 42);
    }
    { // user-allocated memory
        int  count       = -1;
        auto param       = send_recv_count_out(count).construct_buffer_or_rebind();
        using param_type = std::remove_reference_t<decltype(param)>;
        EXPECT_TRUE((std::is_same_v<param_type::value_type, int>));
        EXPECT_EQ(param_type::parameter_type, ParameterType::send_recv_count);
        EXPECT_EQ(param_type::buffer_type, BufferType::out_buffer);
        EXPECT_TRUE(param_type::is_modifiable);
        EXPECT_EQ(param.get_single_element(), -1);
        param.underlying() = 42;
        EXPECT_EQ(param.get_single_element(), 42);
        EXPECT_EQ(count, 42);
    }
}

TEST(ParameterFactoriesTest, root_basics) {
    auto root_obj = root(22);
    EXPECT_EQ(root_obj.rank_signed(), 22);
    EXPECT_EQ(decltype(root_obj)::parameter_type, ParameterType::root);
}

TEST(ParameterFactoriesTest, destination_basics) {
    {
        auto destination_obj = destination(22);
        EXPECT_EQ(destination_obj.rank_signed(), 22);
        EXPECT_EQ(decltype(destination_obj)::parameter_type, ParameterType::destination);
        EXPECT_EQ(decltype(destination_obj)::rank_type, RankType::value);
    }
    {
        auto destination_obj = destination(rank::null);
        EXPECT_EQ(destination_obj.rank_signed(), MPI_PROC_NULL);
        EXPECT_EQ(decltype(destination_obj)::parameter_type, ParameterType::destination);
        EXPECT_EQ(decltype(destination_obj)::rank_type, RankType::null);
    }
}

TEST(ParameterFactoriesTest, source_basics) {
    {
        auto source_obj = source(22);
        EXPECT_EQ(source_obj.rank_signed(), 22);
        EXPECT_EQ(decltype(source_obj)::parameter_type, ParameterType::source);
        EXPECT_EQ(decltype(source_obj)::rank_type, RankType::value);
    }
    {
        auto source_obj = source(rank::null);
        EXPECT_EQ(source_obj.rank_signed(), MPI_PROC_NULL);
        EXPECT_EQ(decltype(source_obj)::parameter_type, ParameterType::source);
        EXPECT_EQ(decltype(source_obj)::rank_type, RankType::null);
    }
    {
        auto source_obj = source(rank::any);
        EXPECT_EQ(source_obj.rank_signed(), MPI_ANY_SOURCE);
        EXPECT_EQ(decltype(source_obj)::parameter_type, ParameterType::source);
        EXPECT_EQ(decltype(source_obj)::rank_type, RankType::any);
    }
}

TEST(ParameterFactoriesTest, tag_basics) {
    {
        auto tag_obj = tag(22);
        EXPECT_EQ(tag_obj.tag(), 22);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = tag(tags::any);
        EXPECT_EQ(tag_obj.tag(), MPI_ANY_TAG);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::any);
    }
}

TEST(ParameterFactoriesTest, tag_enum) {
    enum Tags : int {
        type_a = 27,
        type_b = 3,
    };
    {
        auto tag_obj = tag(Tags::type_a);
        EXPECT_EQ(tag_obj.tag(), 27);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = tag(Tags::type_b);
        EXPECT_EQ(tag_obj.tag(), 3);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
}

TEST(ParameterFactoriesTest, tag_enum_class) {
    enum class Tags {
        type_a = 27,
        type_b = 3,
    };
    {
        auto tag_obj = tag(Tags::type_a);
        EXPECT_EQ(tag_obj.tag(), 27);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = tag(Tags::type_b);
        EXPECT_EQ(tag_obj.tag(), 3);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
}

TEST(ParameterFactoriesTest, send_tag_basics) {
    {
        auto tag_obj = send_tag(22);
        EXPECT_EQ(tag_obj.tag(), 22);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::send_tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = send_tag(tags::any);
        EXPECT_EQ(tag_obj.tag(), MPI_ANY_TAG);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::any);
    }
}

TEST(ParameterFactoriesTest, send_tag_enum) {
    enum Tags : int {
        type_a = 27,
        type_b = 3,
    };
    {
        auto tag_obj = send_tag(Tags::type_a);
        EXPECT_EQ(tag_obj.tag(), 27);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::send_tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = send_tag(Tags::type_b);
        EXPECT_EQ(tag_obj.tag(), 3);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::send_tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
}

TEST(ParameterFactoriesTest, send_tag_enum_class) {
    enum class Tags {
        type_a = 27,
        type_b = 3,
    };
    {
        auto tag_obj = send_tag(Tags::type_a);
        EXPECT_EQ(tag_obj.tag(), 27);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = send_tag(Tags::type_b);
        EXPECT_EQ(tag_obj.tag(), 3);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::send_tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
}

TEST(ParameterFactoriesTest, recv_tag_basics) {
    {
        auto tag_obj = recv_tag(22);
        EXPECT_EQ(tag_obj.tag(), 22);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::recv_tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = recv_tag(tags::any);
        EXPECT_EQ(tag_obj.tag(), MPI_ANY_TAG);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::any);
    }
}

TEST(ParameterFactoriesTest, recv_tag_enum) {
    enum Tags : int {
        type_a = 27,
        type_b = 3,
    };
    {
        auto tag_obj = recv_tag(Tags::type_a);
        EXPECT_EQ(tag_obj.tag(), 27);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::recv_tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = recv_tag(Tags::type_b);
        EXPECT_EQ(tag_obj.tag(), 3);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::recv_tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
}

TEST(ParameterFactoriesTest, recv_tag_enum_class) {
    enum class Tags {
        type_a = 27,
        type_b = 3,
    };
    {
        auto tag_obj = recv_tag(Tags::type_a);
        EXPECT_EQ(tag_obj.tag(), 27);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
    {
        auto tag_obj = recv_tag(Tags::type_b);
        EXPECT_EQ(tag_obj.tag(), 3);
        EXPECT_EQ(decltype(tag_obj)::parameter_type, ParameterType::recv_tag);
        EXPECT_EQ(decltype(tag_obj)::tag_type, TagType::value);
    }
}

TEST(ParameterFactoriesTest, status_basics) {
    {
        auto status_obj = status(kamping::ignore<>).construct_buffer_or_rebind();
        EXPECT_EQ(status_param_to_native_ptr(status_obj), MPI_STATUS_IGNORE);
        EXPECT_EQ(decltype(status_obj)::parameter_type, ParameterType::status);
        EXPECT_EQ(decltype(status_obj)::buffer_type, BufferType::ignore);
    }
    {
        MPI_Status native_status;
        auto       status_obj = status_out(native_status).construct_buffer_or_rebind();
        EXPECT_EQ(status_param_to_native_ptr(status_obj), &native_status);
        EXPECT_EQ(decltype(status_obj)::parameter_type, ParameterType::status);
        EXPECT_TRUE((std::is_same_v<decltype(status_obj)::value_type, MPI_Status>));
        EXPECT_FALSE(decltype(status_obj)::is_owning);
    }
    {
        kamping::Status stat;
        auto            status_obj = status_out(stat).construct_buffer_or_rebind();
        EXPECT_EQ(status_param_to_native_ptr(status_obj), &stat.native());
        EXPECT_EQ(decltype(status_obj)::parameter_type, ParameterType::status);
        EXPECT_TRUE((std::is_same_v<decltype(status_obj)::value_type, Status>));
        EXPECT_FALSE(decltype(status_obj)::is_owning);
    }
    {
        auto status_obj = status_out().construct_buffer_or_rebind();
        EXPECT_EQ(decltype(status_obj)::parameter_type, ParameterType::status);
        EXPECT_TRUE(decltype(status_obj)::is_owning);
        EXPECT_TRUE((std::is_same_v<decltype(status_obj)::value_type, Status>));
        // directly modify the owned status object
        status_param_to_native_ptr(status_obj)->MPI_TAG = 42;
        auto stat                                       = status_obj.extract();
        EXPECT_EQ(stat.tag(), 42);
    }
}

TEST(ParameterFactoriesTest, request_basics) {
    {
        SCOPED_TRACE("owning request");
        auto req_obj = request();
        EXPECT_EQ(req_obj.underlying().mpi_request(), MPI_REQUEST_NULL);
        EXPECT_TRUE(decltype(req_obj)::is_lib_allocated);
        testing::test_single_element_buffer(
            req_obj,
            ParameterType::request,
            BufferType::out_buffer,
            kamping::Request{},
            true /*should_be_modifiable*/
        );
    }
    {
        SCOPED_TRACE("referenced request");
        Request my_request;
        auto    req_obj = request(my_request);
        // check if taken by reference, i.e. this points to the same object
        EXPECT_EQ(&req_obj.underlying(), &my_request);
        EXPECT_EQ(req_obj.underlying().mpi_request(), MPI_REQUEST_NULL);
        EXPECT_FALSE(decltype(req_obj)::is_lib_allocated);
        testing::test_single_element_buffer(
            req_obj,
            ParameterType::request,
            BufferType::out_buffer,
            std::move(my_request),
            true /*should_be_modifiable*/
        );
    }
}

TEST(ParameterFactoriesTest, test_send_mode) {
    ASSERT_TRUE((std::is_same_v<decltype(send_mode(send_modes::standard))::send_mode, internal::standard_mode_t>));
    ASSERT_TRUE((std::is_same_v<decltype(send_mode(send_modes::buffered))::send_mode, internal::buffered_mode_t>));
    ASSERT_TRUE((std::is_same_v<decltype(send_mode(send_modes::synchronous))::send_mode, internal::synchronous_mode_t>)
    );
    ASSERT_TRUE((std::is_same_v<decltype(send_mode(send_modes::ready))::send_mode, internal::ready_mode_t>));
}

TEST(ParameterFactoriesTest, send_recv_buf_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_recv_buf(int_vec).construct_buffer_or_rebind();
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_modifiable_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_recv_buf(const_int_vec).construct_buffer_or_rebind();
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_single_element) {
    {
        uint8_t value                     = 11;
        auto    gen_single_element_buffer = send_recv_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_recv_buf,
            internal::BufferType::in_out_buffer,
            value,
            true
        );
    }
    {
        uint16_t value                     = 4211;
        auto     gen_single_element_buffer = send_recv_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_recv_buf,
            internal::BufferType::in_out_buffer,
            value,
            true
        );
    }
    {
        uint32_t const value                     = 4096;
        auto           gen_single_element_buffer = send_recv_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_recv_buf,
            internal::BufferType::in_out_buffer,
            value,
            false
        );
    }
    {
        uint64_t const value                     = 555555;
        auto           gen_single_element_buffer = send_recv_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_recv_buf,
            internal::BufferType::in_out_buffer,
            value,
            false
        );
    }
    {
        struct CustomType {
            uint64_t v1;
            int      v2;
            char     v3;

            bool operator==(CustomType const& other) const {
                return std::tie(v1, v2, v3) == std::tie(other.v1, other.v2, other.v3);
            }
        }; // struct CustomType
        CustomType value                     = {843290834, -482, 'a'};
        auto       gen_single_element_buffer = send_recv_buf(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_recv_buf,
            internal::BufferType::in_out_buffer,
            value,
            true
        );
    }
}

TEST(ParameterFactoriesTest, single_and_multiple_element_const_send_recv_buffer_type) {
    uint8_t const              value  = 0;
    std::vector<uint8_t> const values = {0, 0, 0, 0, 0, 0};

    auto gen_single_element_buffer = send_recv_buf(value).construct_buffer_or_rebind();
    auto gen_int_vec_buffer        = send_recv_buf(values).construct_buffer_or_rebind();

    bool const single_result = std::is_same_v<
        decltype(gen_single_element_buffer),
        SingleElementConstBuffer<uint8_t, ParameterType::send_recv_buf, BufferType::in_out_buffer>>;
    EXPECT_TRUE(single_result);
    bool const vec_result = std::is_same_v<
        decltype(gen_int_vec_buffer),
        ContainerBasedConstBuffer<std::vector<uint8_t>, ParameterType::send_recv_buf, BufferType::in_out_buffer>>;
    EXPECT_TRUE(vec_result);
}

TEST(ParameterFactoriesTest, single_and_multiple_element_modifiable_send_recv_buffer_type) {
    uint8_t              value  = 0;
    std::vector<uint8_t> values = {0, 0, 0, 0, 0, 0};

    auto gen_single_element_buffer = send_recv_buf(value).construct_buffer_or_rebind();
    auto gen_int_vec_buffer        = send_recv_buf(values).construct_buffer_or_rebind();

    bool const single_result = std::is_same_v<
        decltype(gen_single_element_buffer),
        SingleElementModifiableBuffer<uint8_t, ParameterType::send_recv_buf, BufferType::in_out_buffer>>;
    EXPECT_TRUE(single_result);
    bool const vec_result = std::is_same_v<
        decltype(gen_int_vec_buffer),
        UserAllocatedContainerBasedBuffer<
            std::vector<uint8_t>,
            ParameterType::send_recv_buf,
            BufferType::in_out_buffer,
            BufferResizePolicy::no_resize>>;
    EXPECT_TRUE(vec_result);
}

TEST(ParameterFactoriesTest, send_recv_buf_basics_user_alloc) {
    size_t const     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_on_user_alloc_vector = send_recv_buf(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                      = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer,
        BufferResizePolicy::no_resize,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_send_recv_buf_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::resize_to_fit;
    auto buffer_on_user_alloc_vector       = send_recv_buf<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, resizing_if_required_send_recv_buf_basics_user_alloc) {
    std::vector<int>         int_vec;
    BufferResizePolicy const resize_policy = BufferResizePolicy::grow_only;
    auto buffer_on_user_alloc_vector       = send_recv_buf<resize_policy>(int_vec).construct_buffer_or_rebind();
    using ExpectedValueType                = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer,
        resize_policy,
        int_vec
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = send_recv_buf(alloc_new<std::vector<int>>).construct_buffer_or_rebind();
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_custom_type_library_alloc) {
    struct CustomType {
        uint64_t v1;
        int      v2;
        char     v3;

        bool operator==(CustomType const& other) const {
            return std::tie(v1, v2, v3) == std::tie(other.v1, other.v2, other.v3);
        }
    }; // struct CustomType

    auto buffer_based_on_library_alloc_vector =
        send_recv_buf(alloc_new<std::vector<CustomType>>).construct_buffer_or_rebind();
    using ExpectedValueType = CustomType;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_custom_container_library_alloc) {
    auto buffer_based_on_library_alloc_vector =
        send_recv_buf(alloc_new<testing::OwnContainer<int>>).construct_buffer_or_rebind();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_alloc_container_of_with_own_container) {
    auto buffer_based_on_library_alloc_vector =
        send_recv_buf(alloc_container_of<int>).template construct_buffer_or_rebind<testing::OwnContainer>();
    using ExpectedValueType = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_counts_single_value_in_basics) {
    {
        int  value             = 42;
        auto recv_count_in_obj = recv_counts(value).construct_buffer_or_rebind();
        EXPECT_EQ(*recv_count_in_obj.get().data(), 42);
        EXPECT_FALSE(decltype(recv_count_in_obj)::is_modifiable);
    }

    {
        // passed as rvalue
        auto recv_count_in_obj = recv_counts(42).construct_buffer_or_rebind();
        EXPECT_EQ(*recv_count_in_obj.get().data(), 42);
        EXPECT_FALSE(decltype(recv_count_in_obj)::is_modifiable);
    }
}

TEST(ParameterFactoriesTest, recv_count_out_basics) {
    {
        int  recv_count;
        auto recv_count_out_obj          = recv_counts_out(recv_count).construct_buffer_or_rebind();
        *recv_count_out_obj.get().data() = 42;
        EXPECT_EQ(*recv_count_out_obj.get().data(), 42);
        EXPECT_EQ(recv_count, 42);
        EXPECT_TRUE(decltype(recv_count_out_obj)::is_modifiable);
        EXPECT_EQ(decltype(recv_count_out_obj)::buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto recv_count_out_obj = recv_counts_out().template construct_buffer_or_rebind<std::vector>();
        EXPECT_TRUE(decltype(recv_count_out_obj)::is_modifiable);
        EXPECT_EQ(decltype(recv_count_out_obj)::buffer_type, internal::BufferType::out_buffer);
    }
}

TEST(ParameterFactoriesTest, out_parameter_without_passed_parameters) {
    {
        auto data_buf = recv_counts_out().template construct_buffer_or_rebind<std::vector>();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::recv_counts);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto data_buf = send_displs_out().template construct_buffer_or_rebind<std::vector>();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::send_displs);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto data_buf = recv_counts_out().template construct_buffer_or_rebind<std::vector>();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::recv_counts);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto data_buf = recv_displs_out().template construct_buffer_or_rebind<std::vector>();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::recv_displs);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto data_buf = send_counts_out().template construct_buffer_or_rebind<std::vector>();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::send_counts);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
}

// values_on_rank_0 can never be an out parameter and never be lib allocated, it's always an in parameter.
TEST(ParameterFactoriesTest, values_on_rank_0_single_value_in_basics) {
    {
        int  value         = 42;
        auto values_in_obj = values_on_rank_0(value).construct_buffer_or_rebind();
        EXPECT_EQ(*values_in_obj.get().data(), 42);
        EXPECT_FALSE(decltype(values_in_obj)::is_modifiable);
    }

    {
        // passed as rvalue
        auto values_in_obj = values_on_rank_0(42).construct_buffer_or_rebind();
        EXPECT_EQ(*values_in_obj.get().data(), 42);
        EXPECT_FALSE(decltype(values_in_obj)::is_modifiable);
    }
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = values_on_rank_0(int_vec).construct_buffer_or_rebind();
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = values_on_rank_0(const_int_vec).construct_buffer_or_rebind();
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_referencing_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_moved_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    std::vector<int> const expected          = const_int_vec;
    auto                   gen_via_moved_vec = values_on_rank_0(std::move(const_int_vec)).construct_buffer_or_rebind();
    using ExpectedValueType                  = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_moved_vec,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_vector_from_function) {
    auto make_vector = []() {
        std::vector<int> vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
        return vec;
    };
    std::vector<int> const expected                  = make_vector();
    auto                   gen_via_vec_from_function = values_on_rank_0(make_vector()).construct_buffer_or_rebind();
    using ExpectedValueType                          = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_vec_from_function,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_vector_from_initializer_list) {
    std::vector<int> expected      = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto gen_via_vec_from_function = values_on_rank_0({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1}).construct_buffer_or_rebind();
    using ExpectedValueType        = int;
    testing::test_const_owning_buffer<ExpectedValueType>(
        gen_via_vec_from_function,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_single_element) {
    {
        uint8_t value                     = 11;
        auto    gen_single_element_buffer = values_on_rank_0(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            value
        );
    }
    {
        uint16_t value                     = 4211;
        auto     gen_single_element_buffer = values_on_rank_0(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            value
        );
    }
    {
        uint32_t value                     = 4096;
        auto     gen_single_element_buffer = values_on_rank_0(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            value
        );
    }
    {
        uint64_t value                     = 555555;
        auto     gen_single_element_buffer = values_on_rank_0(value).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            value
        );
    }
    {
        // pass value as rvalue
        auto gen_single_element_buffer = values_on_rank_0(42051).construct_buffer_or_rebind();
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            42051
        );
    }
    {
        struct CustomType {
            uint64_t v1;
            int      v2;
            char     v3;

            bool operator==(CustomType const& other) const {
                return std::tie(v1, v2, v3) == std::tie(other.v1, other.v2, other.v3);
            }
        }; // struct CustomType
        {
            CustomType value                     = {843290834, -482, 'a'};
            auto       gen_single_element_buffer = values_on_rank_0(value).construct_buffer_or_rebind();
            testing::test_single_element_buffer(
                gen_single_element_buffer,
                ParameterType::values_on_rank_0,
                BufferType::in_buffer,
                value
            );
        }
        {
            auto gen_single_element_buffer =
                values_on_rank_0(CustomType{843290834, -482, 'a'}).construct_buffer_or_rebind();
            testing::test_single_element_buffer(
                gen_single_element_buffer,
                ParameterType::values_on_rank_0,
                BufferType::in_buffer,
                CustomType{843290834, -482, 'a'}
            );
        }
    }
}

TEST(ParameterFactoriesTest, values_on_rank_0_switch) {
    uint8_t              value  = 0;
    std::vector<uint8_t> values = {0, 0, 0, 0, 0, 0};

    [[maybe_unused]] auto gen_single_element_buffer        = values_on_rank_0(value).construct_buffer_or_rebind();
    [[maybe_unused]] auto gen_int_vec_buffer               = values_on_rank_0(values).construct_buffer_or_rebind();
    [[maybe_unused]] auto gen_single_element_owning_buffer = values_on_rank_0(uint8_t(0)).construct_buffer_or_rebind();
    [[maybe_unused]] auto gen_int_vec_owning_buffer =
        values_on_rank_0(std::vector<uint8_t>{0, 0, 0, 0, 0, 0}).construct_buffer_or_rebind();

    bool const single_result = std::is_same_v<
        decltype(gen_single_element_buffer),
        SingleElementConstBuffer<uint8_t, ParameterType::values_on_rank_0, BufferType::in_buffer>>;
    EXPECT_TRUE(single_result);
    bool const vec_result = std::is_same_v<
        decltype(gen_int_vec_buffer),
        ContainerBasedConstBuffer<std::vector<uint8_t>, ParameterType::values_on_rank_0, BufferType::in_buffer>>;
    EXPECT_TRUE(vec_result);
    bool const owning_single_result = std::is_same_v<
        decltype(gen_single_element_owning_buffer),
        SingleElementOwningBuffer<uint8_t, ParameterType::values_on_rank_0, BufferType::in_buffer>>;
    EXPECT_TRUE(owning_single_result);
    bool const owning_vec_result = std::is_same_v<
        decltype(gen_int_vec_owning_buffer),
        ContainerBasedOwningBuffer<std::vector<uint8_t>, ParameterType::values_on_rank_0, BufferType::in_buffer>>;
    EXPECT_TRUE(owning_vec_result);
}
