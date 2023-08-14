// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_ops.hpp"

template <typename T>
class TypedOperationsTest : public ::testing::Test {
public:
    using operation_type = T;
};

struct DummyType {
    int  a;
    char b;
};

template <typename Op, typename T>
constexpr T identity_op_value(Op operation, T const& value) {
    auto const identity = kamping::internal::mpi_operation_traits<Op, T>::identity;
    return operation(identity, value);
}

template <typename Op, typename T>
T value_op_identity(Op operation, T const value) {
    auto const& identity = kamping::internal::mpi_operation_traits<Op, T>::identity;
    return operation(value, identity);
}

template <typename T>
struct some_values {
    static std::vector<T> value() {
        return {};
    }
};

template <>
struct some_values<int32_t> {
    static std::vector<int32_t> value() {
        return {std::numeric_limits<int32_t>::lowest(), -1000, -2, 0, 1, 10, 42, std::numeric_limits<int32_t>::max()};
    }
};

template <>
struct some_values<uint32_t> {
    static std::vector<uint32_t> value() {
        return {0, 1, 10, 42, std::numeric_limits<uint32_t>::max()};
    }
};

template <>
struct some_values<int64_t> {
    static std::vector<int64_t> value() {
        // Yes, it's intentional that we're mixing up the values by using some boundaries of a type with less bits.
        return {std::numeric_limits<int64_t>::lowest(), -1000, -2, 0, 1, 10, 42, std::numeric_limits<int32_t>::max()};
    }
};

template <>
struct some_values<uint64_t> {
    static std::vector<uint64_t> value() {
        return {0, 1, 10, 42, std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint64_t>::max()};
    }
};

template <>
struct some_values<float> {
    static std::vector<float> value() {
        return {
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::min(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::epsilon(),
            -1000,
            -2,
            0,
            1,
            10,
            42,
            1.1337f,
            4.2f};
    }
};

template <>
struct some_values<double> {
    static std::vector<double> value() {
        return {
            std::numeric_limits<double>::lowest(),
            std::numeric_limits<double>::min(),
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::epsilon(),
            -1000,
            -2,
            0,
            1,
            10,
            42,
            1.1337,
            4.2};
    }
};

template <>
struct some_values<std::complex<double>> {
    static std::vector<std::complex<double>> value() {
        return {{0, 0}, {1, 1}, {0, -1}, {-1, 0}, {100, -1.34}};
    }
};

template <typename T>
std::vector<T> some_values_v = some_values<T>::value();

using MyTypes =
    ::testing::Types<int32_t, u_int32_t, int64_t, u_int64_t, float, double, std::complex<double>, DummyType>;
TYPED_TEST_SUITE(TypedOperationsTest, MyTypes, );

TYPED_TEST(TypedOperationsTest, test_builtin_operations) {
    using namespace kamping::internal;
    using T = typename TestFixture::operation_type;

    if constexpr (kamping::mpi_type_traits<T>::category == kamping::TypeCategory::integer || kamping::mpi_type_traits<T>::category == kamping::TypeCategory::floating) {
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::max<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::max<T>, T>::op()), MPI_MAX);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::max<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::max<>, T>::op()), MPI_MAX);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::max<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(value, (value_op_identity(kamping::ops::max<T>{}, value)));
            EXPECT_EQ(value, (identity_op_value(kamping::ops::max<T>{}, value)));
        }

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::min<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::min<T>, T>::op()), MPI_MIN);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::min<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::min<>, T>::op()), MPI_MIN);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::min<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(value, (value_op_identity(kamping::ops::min<T>{}, value)));
            EXPECT_EQ(value, (identity_op_value(kamping::ops::min<T>{}, value)));
        }
    }

    if constexpr (kamping::mpi_type_traits<T>::category == kamping::TypeCategory::integer || kamping::mpi_type_traits<T>::category == kamping::TypeCategory::floating || kamping::mpi_type_traits<T>::category == kamping::TypeCategory::complex) {
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::plus<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::plus<T>, T>::op()), MPI_SUM);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::plus<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::plus<>, T>::op()), MPI_SUM);
        // should also work with std::plus
        EXPECT_TRUE((mpi_operation_traits<std::plus<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::plus<>, T>::op()), MPI_SUM);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::plus<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(value, (value_op_identity(kamping::ops::plus<T>{}, value)));
            EXPECT_EQ(value, (identity_op_value(kamping::ops::plus<T>{}, value)));
        }

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::multiplies<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::multiplies<T>, T>::op()), MPI_PROD);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::multiplies<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::multiplies<>, T>::op()), MPI_PROD);
        EXPECT_TRUE((mpi_operation_traits<std::multiplies<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::multiplies<>, T>::op()), MPI_PROD);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::multiplies<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(value, (value_op_identity(kamping::ops::multiplies<T>{}, value)));
            EXPECT_EQ(value, (identity_op_value(kamping::ops::multiplies<T>{}, value)));
        }
    }

    if constexpr (kamping::mpi_type_traits<T>::category == kamping::TypeCategory::integer || kamping::mpi_type_traits<T>::category == kamping::TypeCategory::logical) {
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_and<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_and<T>, T>::op()), MPI_LAND);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_and<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_and<>, T>::op()), MPI_LAND);
        EXPECT_TRUE((mpi_operation_traits<std::logical_and<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::logical_and<>, T>::op()), MPI_LAND);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::logical_and<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(static_cast<bool>(value), (value_op_identity(kamping::ops::logical_and<T>{}, value)));
            EXPECT_EQ(static_cast<bool>(value), (identity_op_value(kamping::ops::logical_and<T>{}, value)));
        }

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_or<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_or<T>, T>::op()), MPI_LOR);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_or<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_or<>, T>::op()), MPI_LOR);
        EXPECT_TRUE((mpi_operation_traits<std::logical_or<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::logical_or<>, T>::op()), MPI_LOR);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::logical_or<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(static_cast<bool>(value), (value_op_identity(kamping::ops::logical_or<T>{}, value)));
            EXPECT_EQ(static_cast<bool>(value), (identity_op_value(kamping::ops::logical_or<T>{}, value)));
        }

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_xor<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_xor<T>, T>::op()), MPI_LXOR);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_xor<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_xor<>, T>::op()), MPI_LXOR);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::logical_xor<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(static_cast<bool>(value), (value_op_identity(kamping::ops::logical_xor<T>{}, value)));
            EXPECT_EQ(static_cast<bool>(value), (identity_op_value(kamping::ops::logical_xor<T>{}, value)));
        }
    }

    if constexpr (kamping::mpi_type_traits<T>::category == kamping::TypeCategory::integer || kamping::mpi_type_traits<T>::category == kamping::TypeCategory::byte) {
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_and<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_and<T>, T>::op()), MPI_BAND);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_and<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_and<>, T>::op()), MPI_BAND);
        EXPECT_TRUE((mpi_operation_traits<std::bit_and<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::bit_and<>, T>::op()), MPI_BAND);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::bit_and<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(value, (value_op_identity(kamping::ops::bit_and<T>{}, value)));
            EXPECT_EQ(value, (identity_op_value(kamping::ops::bit_and<T>{}, value)));
        }

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_or<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_or<T>, T>::op()), MPI_BOR);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_or<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_or<>, T>::op()), MPI_BOR);
        EXPECT_TRUE((mpi_operation_traits<std::bit_or<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::bit_or<>, T>::op()), MPI_BOR);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::bit_or<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(value, (value_op_identity(kamping::ops::bit_or<T>{}, value)));
            EXPECT_EQ(value, (identity_op_value(kamping::ops::bit_or<T>{}, value)));
        }

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_xor<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_xor<T>, T>::op()), MPI_BXOR);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_xor<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_xor<>, T>::op()), MPI_BXOR);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::bit_xor<std::complex<int>>, T>::is_builtin));
        for (auto& value: some_values_v<T>) {
            EXPECT_EQ(value, (value_op_identity(kamping::ops::bit_xor<T>{}, value)));
            EXPECT_EQ(value, (identity_op_value(kamping::ops::bit_xor<T>{}, value)));
        }
    }
}

TYPED_TEST(TypedOperationsTest, user_defined_operation_is_not_builtin_lambda) {
    using T = typename TestFixture::operation_type;
    auto op = [](auto a, auto b [[maybe_unused]]) {
        return a;
    };
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<decltype(op), T>::is_builtin));
}

TYPED_TEST(TypedOperationsTest, user_defined_operation_is_not_builtin_function_object) {
    using T = typename TestFixture::operation_type;
    struct MyOperation {
        T const& operator()(T const& a, T const& b [[maybe_unused]]) {
            return a;
        }
    };
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<MyOperation, T>::is_builtin));
}

TYPED_TEST(TypedOperationsTest, user_defined_operation_is_not_builtin_unsupported_stl_operation) {
    using T = typename TestFixture::operation_type;
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<std::minus<>, T>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<std::divides<>, T>::is_builtin));
}

TEST(OperationsTest, builtin_operations_on_unsupported_type) {
    // maximum/minimum
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::max<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::max<>, std::complex<double>>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::min<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::min<>, std::complex<double>>::is_builtin));
    // addition/multiplication
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::plus<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::plus<>, bool>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::multiplies<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::multiplies<>, bool>::is_builtin));
    // logical operations
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::logical_and<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::logical_and<>, double>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::logical_or<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::logical_or<>, double>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::logical_xor<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::logical_xor<>, double>::is_builtin));
    // bitwise operations
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::bit_and<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::bit_and<>, double>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::bit_or<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::bit_or<>, double>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::bit_xor<>, DummyType>::is_builtin));
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<kamping::ops::bit_xor<>, double>::is_builtin));
}

TEST(OperationsTest, with_operation_functor) {
    kamping::internal::with_operation_functor(MPI_MAX, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::max<>>));
    });
    kamping::internal::with_operation_functor(MPI_MIN, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::min<>>));
    });
    kamping::internal::with_operation_functor(MPI_SUM, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::plus<>>));
    });
    kamping::internal::with_operation_functor(MPI_PROD, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::multiplies<>>));
    });
    kamping::internal::with_operation_functor(MPI_LAND, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::logical_and<>>));
    });
    kamping::internal::with_operation_functor(MPI_LOR, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::logical_or<>>));
    });
    kamping::internal::with_operation_functor(MPI_LXOR, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::logical_xor<>>));
    });
    kamping::internal::with_operation_functor(MPI_BAND, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::bit_and<>>));
    });
    kamping::internal::with_operation_functor(MPI_BOR, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::bit_or<>>));
    });
    kamping::internal::with_operation_functor(MPI_BXOR, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::bit_xor<>>));
    });
    kamping::internal::with_operation_functor(MPI_OP_NULL, [](auto functor) {
        KASSERT((std::is_same_v<decltype(functor), kamping::ops::null<>>));
    });
}
