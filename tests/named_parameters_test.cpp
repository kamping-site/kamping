// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
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

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "legacy_parameter_objects.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

namespace testing {
template <typename ExpectedValueType, typename GeneratedBuffer, typename T>
void test_const_buffer(
    GeneratedBuffer const&           generated_buffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    Span<T>&                         expected_span
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

TEST(ParameterFactoriesTest, send_buf_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_buf(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_buf_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_buf(const_int_vec);
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_buf_basics_moved_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    std::vector<int> const expected          = const_int_vec;
    auto                   gen_via_moved_vec = send_buf(std::move(const_int_vec));
    using ExpectedValueType                  = int;
    testing::test_owning_buffer<ExpectedValueType>(
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
    auto                   gen_via_vec_from_function = send_buf(make_vector());
    using ExpectedValueType                          = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_vec_from_function,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_buf_basics_vector_from_initializer_list) {
    std::vector<int> expected                  = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_vec_from_function = send_buf({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
    using ExpectedValueType                    = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_vec_from_function,
        ParameterType::send_buf,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_buf_single_element) {
    {
        uint8_t value                     = 11;
        auto    gen_single_element_buffer = send_buf(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            value
        );
    }
    {
        uint16_t value                     = 4211;
        auto     gen_single_element_buffer = send_buf(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            value
        );
    }
    {
        uint32_t value                     = 4096;
        auto     gen_single_element_buffer = send_buf(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            value
        );
    }
    {
        uint64_t value                     = 555555;
        auto     gen_single_element_buffer = send_buf(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_buf,
            internal::BufferType::in_buffer,
            value
        );
    }
    {
        // pass value as rvalue
        auto gen_single_element_buffer = send_buf(42051);
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
            auto       gen_single_element_buffer = send_buf(value);
            testing::test_single_element_buffer(
                gen_single_element_buffer,
                ParameterType::send_buf,
                internal::BufferType::in_buffer,
                value
            );
        }
        {
            auto gen_single_element_buffer = send_buf(CustomType{843290834, -482, 'a'});
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

    [[maybe_unused]] auto gen_single_element_buffer        = send_buf(value);
    [[maybe_unused]] auto gen_int_vec_buffer               = send_buf(values);
    [[maybe_unused]] auto gen_single_element_owning_buffer = send_buf(uint8_t(0));
    [[maybe_unused]] auto gen_int_vec_owning_buffer        = send_buf(std::vector<uint8_t>{0, 0, 0, 0, 0, 0});

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
    auto ignored_send_buf = send_buf(ignore<int>);
    EXPECT_EQ(ignored_send_buf.get().data(), nullptr);
    EXPECT_EQ(ignored_send_buf.get().size(), 0);
}

TEST(ParameterFactoriesTest, send_counts_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_counts(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_counts_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_counts(const_int_vec);
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_counts_basics_moved_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             expected        = int_vec;
    auto             gen_via_int_vec = send_counts(std::move(int_vec));
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_counts_basics_initializer_list) {
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_initializer_list = send_counts({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
    using ExpectedValueType                       = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_int_initializer_list,
        ParameterType::send_counts,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, recv_counts_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = recv_counts(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::recv_counts,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, recv_counts_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = recv_counts(const_int_vec);
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::recv_counts,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, recv_counts_in_basics_moved_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             expected          = int_vec;
    auto             gen_via_moved_vec = recv_counts(std::move(int_vec));
    using ExpectedValueType            = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_moved_vec,
        ParameterType::recv_counts,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, recv_counts_in_basics_initializer_list) {
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_initializer_list = recv_counts({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
    using ExpectedValueType                   = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_initializer_list,
        ParameterType::recv_counts,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_displs_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_displs(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_displs_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = send_displs(const_int_vec);
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_displs_in_basics_moved_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             expected          = int_vec;
    auto             gen_via_moved_vec = send_displs(std::move(int_vec));
    using ExpectedValueType            = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_moved_vec,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, send_displs_in_basics_initializer_list) {
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_initializer_list = send_displs({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
    using ExpectedValueType                   = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_initializer_list,
        ParameterType::send_displs,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, recv_displs_in_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = recv_displs(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::recv_displs,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, recv_displs_in_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = recv_displs(const_int_vec);
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::recv_displs,
        internal::BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, recv_displs_in_basics_moved_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             expected          = int_vec;
    auto             gen_via_moved_vec = recv_displs(std::move(int_vec));
    using ExpectedValueType            = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_moved_vec,
        ParameterType::recv_displs,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, recv_displs_in_basics_initializer_list) {
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_initializer_list = recv_displs({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
    using ExpectedValueType                   = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_initializer_list,
        ParameterType::recv_displs,
        internal::BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, recv_buf_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_on_user_alloc_vector = recv_buf(int_vec);
    using ExpectedValueType                      = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector,
        ParameterType::recv_buf,
        internal::BufferType::out_buffer,
        int_vec
    );
}

TEST(ParameterFactoriesTest, recv_buf_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = recv_buf(NewContainer<std::vector<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_buf,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, send_displs_out_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_vector = send_displs_out(int_vec);
    using ExpectedValueType                            = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::send_displs,
        internal::BufferType::out_buffer,
        int_vec
    );
}

TEST(ParameterFactoriesTest, send_displs_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = send_displs_out(NewContainer<std::vector<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_displs,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_counts_out_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_buffer    = recv_counts_out(int_vec);
    auto             buffer_based_on_library_alloc_vector = recv_counts_out(NewContainer<std::vector<int>>{});
    using ExpectedValueType                               = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_buffer,
        ParameterType::recv_counts,
        internal::BufferType::out_buffer,
        int_vec
    );
}

TEST(ParameterFactoriesTest, recv_counts_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = recv_counts_out(NewContainer<std::vector<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_counts,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_displs_out_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_based_on_user_alloc_vector = recv_displs_out(int_vec);
    using ExpectedValueType                            = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_based_on_user_alloc_vector,
        ParameterType::recv_displs,
        internal::BufferType::out_buffer,
        int_vec
    );
}

TEST(ParameterFactoriesTest, recv_displs_out_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = recv_displs_out(NewContainer<std::vector<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::recv_displs,
        internal::BufferType::out_buffer
    );
}

TEST(ParameterFactoriesTest, root_basics) {
    auto root_obj = root(22);
    EXPECT_EQ(root_obj.rank(), 22);
}

TEST(ParameterFactoriesTest, send_recv_buf_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = send_recv_buf(int_vec);
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
    auto                   gen_via_const_int_vec = send_recv_buf(const_int_vec);
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_single_element) {
    {
        uint8_t value                     = 11;
        auto    gen_single_element_buffer = send_recv_buf(value);
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
        auto     gen_single_element_buffer = send_recv_buf(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_recv_buf,
            internal::BufferType::in_out_buffer,
            value,
            true
        );
    }
    {
        const uint32_t value                     = 4096;
        auto           gen_single_element_buffer = send_recv_buf(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::send_recv_buf,
            internal::BufferType::in_out_buffer,
            value,
            false
        );
    }
    {
        const uint64_t value                     = 555555;
        auto           gen_single_element_buffer = send_recv_buf(value);
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
        auto       gen_single_element_buffer = send_recv_buf(value);
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
    const uint8_t              value  = 0;
    const std::vector<uint8_t> values = {0, 0, 0, 0, 0, 0};

    auto gen_single_element_buffer = send_recv_buf(value);
    auto gen_int_vec_buffer        = send_recv_buf(values);

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

    auto gen_single_element_buffer = send_recv_buf(value);
    auto gen_int_vec_buffer        = send_recv_buf(values);

    bool const single_result = std::is_same_v<
        decltype(gen_single_element_buffer),
        SingleElementModifiableBuffer<uint8_t, ParameterType::send_recv_buf, BufferType::in_out_buffer>>;
    EXPECT_TRUE(single_result);
    bool const vec_result = std::is_same_v<
        decltype(gen_int_vec_buffer),
        UserAllocatedContainerBasedBuffer<
            std::vector<uint8_t>,
            ParameterType::send_recv_buf,
            BufferType::in_out_buffer>>;
    EXPECT_TRUE(vec_result);
}

TEST(ParameterFactoriesTest, send_recv_buf_basics_user_alloc) {
    const size_t     size = 10;
    std::vector<int> int_vec(size);
    auto             buffer_on_user_alloc_vector = send_recv_buf(int_vec);
    using ExpectedValueType                      = int;
    testing::test_user_allocated_buffer<ExpectedValueType>(
        buffer_on_user_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer,
        int_vec
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_basics_library_alloc) {
    auto buffer_based_on_library_alloc_vector = send_recv_buf(NewContainer<std::vector<int>>{});
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

    auto buffer_based_on_library_alloc_vector = send_recv_buf(NewContainer<std::vector<CustomType>>{});
    using ExpectedValueType                   = CustomType;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer
    );
}

TEST(ParameterFactoriesTest, send_recv_buf_custom_container_library_alloc) {
    auto buffer_based_on_library_alloc_vector = send_recv_buf(NewContainer<testing::OwnContainer<int>>{});
    using ExpectedValueType                   = int;
    testing::test_library_allocated_buffer<ExpectedValueType>(
        buffer_based_on_library_alloc_vector,
        ParameterType::send_recv_buf,
        internal::BufferType::in_out_buffer
    );
}

TEST(ParameterFactoriesTest, recv_counts_single_value_in_basics) {
    {
        int  value             = 42;
        auto recv_count_in_obj = recv_counts(value);
        EXPECT_EQ(*recv_count_in_obj.get().data(), 42);
        EXPECT_FALSE(decltype(recv_count_in_obj)::is_modifiable);
    }

    {
        // passed as rvalue
        auto recv_count_in_obj = recv_counts(42);
        EXPECT_EQ(*recv_count_in_obj.get().data(), 42);
        EXPECT_FALSE(decltype(recv_count_in_obj)::is_modifiable);
    }
}

TEST(ParameterFactoriesTest, recv_count_out_basics) {
    {
        int  recv_count;
        auto recv_count_out_obj          = recv_counts_out(recv_count);
        *recv_count_out_obj.get().data() = 42;
        EXPECT_EQ(*recv_count_out_obj.get().data(), 42);
        EXPECT_EQ(recv_count, 42);
        EXPECT_TRUE(decltype(recv_count_out_obj)::is_modifiable);
        EXPECT_EQ(decltype(recv_count_out_obj)::buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto recv_count_out_obj = recv_counts_out();
        EXPECT_TRUE(decltype(recv_count_out_obj)::is_modifiable);
        EXPECT_EQ(decltype(recv_count_out_obj)::buffer_type, internal::BufferType::out_buffer);
    }
}

TEST(ParameterFactoriesTest, recv_counts_out_single_value_lib_allocated_basics) {
    auto recv_count_out_obj          = recv_counts_out(NewContainer<int>{});
    *recv_count_out_obj.get().data() = 42;
    EXPECT_EQ(*recv_count_out_obj.get().data(), 42);
    EXPECT_TRUE(decltype(recv_count_out_obj)::is_modifiable);
    EXPECT_TRUE(has_extract_v<decltype(recv_count_out_obj)>);
}

TEST(ParameterFactoriesTest, make_data_buffer) {
    {
        // Constant, container, referencing, user allocated
        std::vector<int>                  vec;
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::constant, btype>(vec);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
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
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::modifiable, btype>(vec);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
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
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::constant, btype>(single_int);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_TRUE(data_buf.is_single_element);
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
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::constant, btype>(std::move(vec));
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int> const>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }

    {
        // modifiable, container, owning, library allocated
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto                              data_buf =
            internal::make_data_buffer<ptype, BufferModifiability::modifiable, btype>(NewContainer<std::vector<int>>{});
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int>>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, single element, owning, lib_allocated
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::modifiable, btype>(NewContainer<int>{});
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_TRUE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, int>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, container, owning, user_allocated with initializer_list
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::modifiable, btype>({1, 2, 3});
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<int>>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, container, owning, user_allocated with initializer_list
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::constant, btype>({1, 2, 3});
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, const std::vector<int>>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
}

TEST(ParameterFactoriesTest, make_data_buffer_boolean_value) {
    // use a custom container, because std::vector<bool> is not supported (see compilation failure tests)
    {
        // Constant, container, referencing, user allocated
        testing::OwnContainer<bool>       vec   = {true, false};
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::constant, btype>(vec);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        // As this buffer is referencing, the addresses of vec ad data_buf.underlying() should be the same.
        EXPECT_EQ(&vec, &data_buf.underlying());
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, testing::OwnContainer<bool> const&>,
            "Referencing buffers must hold a reference to their data."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, container, referencing, user allocated
        testing::OwnContainer<bool>       vec   = {true, false};
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::modifiable, btype>(vec);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        // As this buffer is referencing, the addresses of vec ad data_buf.underlying() should be the same.
        EXPECT_EQ(&vec, &data_buf.underlying());
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, testing::OwnContainer<bool>&>,
            "Referencing buffers must hold a reference to their data."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, single element, referencing, user allocated
        bool                              single_bool;
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::constant, btype>(single_bool);
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_TRUE(data_buf.is_single_element);
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
        testing::OwnContainer<bool>       vec   = {true, false};
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::constant, btype>(std::move(vec));
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, testing::OwnContainer<bool> const>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }

    {
        // modifiable, container, owning, library allocated
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::modifiable, btype>(
            NewContainer<testing::OwnContainer<bool>>{}
        );
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, testing::OwnContainer<bool>>,
            "Owning buffers must hold their data directly."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_TRUE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Modifiable, single element, owning, lib_allocated
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::modifiable, btype>(NewContainer<bool>{});
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_TRUE(data_buf.is_single_element);
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
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::modifiable, btype>({true, false, true});
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_TRUE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, std::vector<kabool>>,
            "Initializer lists of type bool have to be converted to std::vector<kabool>."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
    {
        // Constant, container, owning, user_allocated with initializer_list
        constexpr internal::ParameterType ptype = internal::ParameterType::send_buf;
        constexpr internal::BufferType    btype = internal::BufferType::in_buffer;
        auto data_buf = internal::make_data_buffer<ptype, BufferModifiability::constant, btype>({true, false, true});
        EXPECT_EQ(data_buf.parameter_type, ptype);
        EXPECT_FALSE(data_buf.is_modifiable);
        EXPECT_FALSE(data_buf.is_single_element);
        static_assert(
            std::is_same_v<decltype(data_buf)::MemberTypeWithConstAndRef, const std::vector<kabool>>,
            "Initializer lists of type bool have to be converted to std::vector<kabool>."
        );
        // extract() as proxy for lib allocated DataBuffers
        EXPECT_FALSE(has_extract_v<decltype(data_buf)>);
    }
}

TEST(ParameterFactoriesTest, out_parameter_without_passed_parameters) {
    {
        auto data_buf = recv_counts_out();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::recv_counts);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto data_buf = send_displs_out();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::send_displs);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto data_buf = recv_counts_out();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::recv_counts);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
    {
        auto data_buf = recv_displs_out();
        EXPECT_EQ(data_buf.parameter_type, internal::ParameterType::recv_displs);
        EXPECT_EQ(data_buf.is_modifiable, true);
        EXPECT_EQ(data_buf.buffer_type, internal::BufferType::out_buffer);
    }
}

// values_on_rank_0 can never be an out parameter and never be lib allocated, it's always an in parameter.
TEST(ParameterFactoriesTest, values_on_rank_0_single_value_in_basics) {
    {
        int  value         = 42;
        auto values_in_obj = values_on_rank_0(value);
        EXPECT_EQ(*values_in_obj.get().data(), 42);
        EXPECT_FALSE(decltype(values_in_obj)::is_modifiable);
    }

    {
        // passed as rvalue
        auto values_in_obj = values_on_rank_0(42);
        EXPECT_EQ(*values_in_obj.get().data(), 42);
        EXPECT_FALSE(decltype(values_in_obj)::is_modifiable);
    }
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_int_vector) {
    std::vector<int> int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_int_vec = values_on_rank_0(int_vec);
    Span<int>        expected_span{int_vec.data(), int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_int_vec,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_const_int_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto                   gen_via_const_int_vec = values_on_rank_0(const_int_vec);
    Span<int const>        expected_span{const_int_vec.data(), const_int_vec.size()};
    using ExpectedValueType = int;
    testing::test_const_buffer<ExpectedValueType>(
        gen_via_const_int_vec,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected_span
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_moved_vector) {
    std::vector<int> const const_int_vec{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    std::vector<int> const expected          = const_int_vec;
    auto                   gen_via_moved_vec = values_on_rank_0(std::move(const_int_vec));
    using ExpectedValueType                  = int;
    testing::test_owning_buffer<ExpectedValueType>(
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
    auto                   gen_via_vec_from_function = values_on_rank_0(make_vector());
    using ExpectedValueType                          = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_vec_from_function,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_basics_vector_from_initializer_list) {
    std::vector<int> expected                  = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto             gen_via_vec_from_function = values_on_rank_0({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
    using ExpectedValueType                    = int;
    testing::test_owning_buffer<ExpectedValueType>(
        gen_via_vec_from_function,
        ParameterType::values_on_rank_0,
        BufferType::in_buffer,
        expected
    );
}

TEST(ParameterFactoriesTest, values_on_rank_0_single_element) {
    {
        uint8_t value                     = 11;
        auto    gen_single_element_buffer = values_on_rank_0(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            value
        );
    }
    {
        uint16_t value                     = 4211;
        auto     gen_single_element_buffer = values_on_rank_0(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            value
        );
    }
    {
        uint32_t value                     = 4096;
        auto     gen_single_element_buffer = values_on_rank_0(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            value
        );
    }
    {
        uint64_t value                     = 555555;
        auto     gen_single_element_buffer = values_on_rank_0(value);
        testing::test_single_element_buffer(
            gen_single_element_buffer,
            ParameterType::values_on_rank_0,
            BufferType::in_buffer,
            value
        );
    }
    {
        // pass value as rvalue
        auto gen_single_element_buffer = values_on_rank_0(42051);
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
            auto       gen_single_element_buffer = values_on_rank_0(value);
            testing::test_single_element_buffer(
                gen_single_element_buffer,
                ParameterType::values_on_rank_0,
                BufferType::in_buffer,
                value
            );
        }
        {
            auto gen_single_element_buffer = values_on_rank_0(CustomType{843290834, -482, 'a'});
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

    [[maybe_unused]] auto gen_single_element_buffer        = values_on_rank_0(value);
    [[maybe_unused]] auto gen_int_vec_buffer               = values_on_rank_0(values);
    [[maybe_unused]] auto gen_single_element_owning_buffer = values_on_rank_0(uint8_t(0));
    [[maybe_unused]] auto gen_int_vec_owning_buffer        = values_on_rank_0(std::vector<uint8_t>{0, 0, 0, 0, 0, 0});

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