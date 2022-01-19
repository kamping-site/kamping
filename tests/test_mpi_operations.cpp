#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <sys/types.h>

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

using MyTypes =
    ::testing::Types<int32_t, u_int32_t, int64_t, u_int64_t, float, double, std::complex<double>, DummyType>;
TYPED_TEST_SUITE(TypedOperationsTest, MyTypes, );

TEST(TypeGroupsTest, test_type_groups) {
    EXPECT_TRUE(kamping::mpi_type_traits<int32_t>::is_integer);
    EXPECT_FALSE(kamping::mpi_type_traits<int32_t>::is_float);
    EXPECT_FALSE(kamping::mpi_type_traits<int32_t>::is_logical);
    EXPECT_FALSE(kamping::mpi_type_traits<int32_t>::is_complex);
    EXPECT_FALSE(kamping::mpi_type_traits<int32_t>::is_byte);

    EXPECT_TRUE(kamping::mpi_type_traits<u_int32_t>::is_integer);
    EXPECT_FALSE(kamping::mpi_type_traits<u_int32_t>::is_float);
    EXPECT_FALSE(kamping::mpi_type_traits<u_int32_t>::is_logical);
    EXPECT_FALSE(kamping::mpi_type_traits<u_int32_t>::is_complex);
    EXPECT_FALSE(kamping::mpi_type_traits<u_int32_t>::is_byte);

    EXPECT_TRUE(kamping::mpi_type_traits<int64_t>::is_integer);
    EXPECT_FALSE(kamping::mpi_type_traits<int64_t>::is_float);
    EXPECT_FALSE(kamping::mpi_type_traits<int64_t>::is_logical);
    EXPECT_FALSE(kamping::mpi_type_traits<int64_t>::is_complex);
    EXPECT_FALSE(kamping::mpi_type_traits<int64_t>::is_byte);

    EXPECT_TRUE(kamping::mpi_type_traits<u_int64_t>::is_integer);
    EXPECT_FALSE(kamping::mpi_type_traits<u_int64_t>::is_float);
    EXPECT_FALSE(kamping::mpi_type_traits<u_int64_t>::is_logical);
    EXPECT_FALSE(kamping::mpi_type_traits<u_int64_t>::is_complex);
    EXPECT_FALSE(kamping::mpi_type_traits<u_int64_t>::is_byte);

    EXPECT_FALSE(kamping::mpi_type_traits<float>::is_integer);
    EXPECT_TRUE(kamping::mpi_type_traits<float>::is_float);
    EXPECT_FALSE(kamping::mpi_type_traits<float>::is_logical);
    EXPECT_FALSE(kamping::mpi_type_traits<float>::is_complex);
    EXPECT_FALSE(kamping::mpi_type_traits<float>::is_byte);

    EXPECT_FALSE(kamping::mpi_type_traits<double>::is_integer);
    EXPECT_TRUE(kamping::mpi_type_traits<double>::is_float);
    EXPECT_FALSE(kamping::mpi_type_traits<double>::is_logical);
    EXPECT_FALSE(kamping::mpi_type_traits<double>::is_complex);
    EXPECT_FALSE(kamping::mpi_type_traits<double>::is_byte);

    EXPECT_FALSE(kamping::mpi_type_traits<std::complex<double>>::is_integer);
    EXPECT_FALSE(kamping::mpi_type_traits<std::complex<double>>::is_float);
    EXPECT_FALSE(kamping::mpi_type_traits<std::complex<double>>::is_logical);
    EXPECT_TRUE(kamping::mpi_type_traits<std::complex<double>>::is_complex);
    EXPECT_FALSE(kamping::mpi_type_traits<std::complex<double>>::is_byte);

    EXPECT_FALSE(kamping::mpi_type_traits<DummyType>::is_integer);
    EXPECT_FALSE(kamping::mpi_type_traits<DummyType>::is_float);
    EXPECT_FALSE(kamping::mpi_type_traits<DummyType>::is_logical);
    EXPECT_FALSE(kamping::mpi_type_traits<DummyType>::is_complex);
    EXPECT_FALSE(kamping::mpi_type_traits<DummyType>::is_byte);
}


TYPED_TEST(TypedOperationsTest, test_builtin_operations) {
    using namespace kamping::internal;
    using T = typename TestFixture::operation_type;

    if constexpr (kamping::mpi_type_traits<T>::is_integer || kamping::mpi_type_traits<T>::is_float) {
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::max<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::max<T>, T>::op()), MPI_MAX);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::max<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::max<>, T>::op()), MPI_MAX);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::max<std::complex<int>>, T>::value));

        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::min<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::min<T>, T>::op()), MPI_MIN);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::min<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::min<>, T>::op()), MPI_MIN);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::min<std::complex<int>>, T>::value));
    }

    if constexpr (
        kamping::mpi_type_traits<T>::is_integer || kamping::mpi_type_traits<T>::is_float
        || kamping::mpi_type_traits<T>::is_complex) {
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::plus<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::plus<T>, T>::op()), MPI_SUM);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::plus<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::plus<>, T>::op()), MPI_SUM);
        // should also work with std::plus
        EXPECT_TRUE((is_builtin_mpi_op<std::plus<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<std::plus<>, T>::op()), MPI_SUM);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::plus<std::complex<int>>, T>::value));

        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::multiplies<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::multiplies<T>, T>::op()), MPI_PROD);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::multiplies<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::multiplies<>, T>::op()), MPI_PROD);
        EXPECT_TRUE((is_builtin_mpi_op<std::multiplies<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<std::multiplies<>, T>::op()), MPI_PROD);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::multiplies<std::complex<int>>, T>::value));
    }

    if constexpr (kamping::mpi_type_traits<T>::is_integer || kamping::mpi_type_traits<T>::is_logical) {
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::logical_and<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::logical_and<T>, T>::op()), MPI_LAND);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::logical_and<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::logical_and<>, T>::op()), MPI_LAND);
        EXPECT_TRUE((is_builtin_mpi_op<std::logical_and<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<std::logical_and<>, T>::op()), MPI_LAND);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::logical_and<std::complex<int>>, T>::value));

        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::logical_or<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::logical_or<T>, T>::op()), MPI_LOR);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::logical_or<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::logical_or<>, T>::op()), MPI_LOR);
        EXPECT_TRUE((is_builtin_mpi_op<std::logical_or<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<std::logical_or<>, T>::op()), MPI_LOR);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::logical_or<std::complex<int>>, T>::value));

        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::logical_xor<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::logical_xor<T>, T>::op()), MPI_LXOR);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::logical_xor<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::logical_xor<>, T>::op()), MPI_LXOR);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::logical_xor<std::complex<int>>, T>::value));
    }

    if constexpr (kamping::mpi_type_traits<T>::is_integer || kamping::mpi_type_traits<T>::is_byte) {
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::bit_and<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::bit_and<T>, T>::op()), MPI_BAND);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::bit_and<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::bit_and<>, T>::op()), MPI_BAND);
        EXPECT_TRUE((is_builtin_mpi_op<std::bit_and<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<std::bit_and<>, T>::op()), MPI_BAND);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::bit_and<std::complex<int>>, T>::value));

        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::bit_or<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::bit_or<T>, T>::op()), MPI_BOR);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::bit_or<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::bit_or<>, T>::op()), MPI_BOR);
        EXPECT_TRUE((is_builtin_mpi_op<std::bit_or<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<std::bit_or<>, T>::op()), MPI_BOR);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::bit_or<std::complex<int>>, T>::value));

        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::bit_xor<T>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::bit_xor<T>, T>::op()), MPI_BXOR);
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::bit_xor<>, T>::value));
        EXPECT_EQ((is_builtin_mpi_op<kamping::ops::bit_xor<>, T>::op()), MPI_BXOR);
        EXPECT_FALSE((is_builtin_mpi_op<kamping::ops::bit_xor<std::complex<int>>, T>::value));
    }
}

TYPED_TEST(TypedOperationsTest, user_defined_operation_is_not_builtin) {
    using T = typename TestFixture::operation_type;
    auto op = [](auto a, auto b [[maybe_unused]]) {
        return a;
    };
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<decltype(op), T>::value));
}


TEST(OperationsTest, builtin_operations_on_unsupported_type) {
    // maximum/minimum
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::max<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::max<>, std::complex<double>>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::min<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::min<>, std::complex<double>>::value));
    // addition/multiplication
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::plus<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::plus<>, bool>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::multiplies<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::multiplies<>, bool>::value));
    // logical operations
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::logical_and<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::logical_and<>, double>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::logical_or<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::logical_or<>, double>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::logical_xor<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::logical_xor<>, double>::value));
    // bitwise operations
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::bit_and<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::bit_and<>, double>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::bit_or<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::bit_or<>, double>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::bit_xor<>, DummyType>::value));
    EXPECT_FALSE((kamping::internal::is_builtin_mpi_op<kamping::ops::bit_xor<>, double>::value));
}
