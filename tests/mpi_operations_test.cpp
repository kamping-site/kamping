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
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::max<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::max<T>, T>::op()), MPI_MAX);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::max<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::max<>, T>::op()), MPI_MAX);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::max<std::complex<int>>, T>::is_builtin));

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::min<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::min<T>, T>::op()), MPI_MIN);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::min<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::min<>, T>::op()), MPI_MIN);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::min<std::complex<int>>, T>::is_builtin));
    }

    if constexpr (
        kamping::mpi_type_traits<T>::is_integer || kamping::mpi_type_traits<T>::is_float
        || kamping::mpi_type_traits<T>::is_complex) {
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::plus<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::plus<T>, T>::op()), MPI_SUM);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::plus<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::plus<>, T>::op()), MPI_SUM);
        // should also work with std::plus
        EXPECT_TRUE((mpi_operation_traits<std::plus<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::plus<>, T>::op()), MPI_SUM);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::plus<std::complex<int>>, T>::is_builtin));

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::multiplies<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::multiplies<T>, T>::op()), MPI_PROD);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::multiplies<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::multiplies<>, T>::op()), MPI_PROD);
        EXPECT_TRUE((mpi_operation_traits<std::multiplies<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::multiplies<>, T>::op()), MPI_PROD);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::multiplies<std::complex<int>>, T>::is_builtin));
    }

    if constexpr (kamping::mpi_type_traits<T>::is_integer || kamping::mpi_type_traits<T>::is_logical) {
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_and<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_and<T>, T>::op()), MPI_LAND);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_and<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_and<>, T>::op()), MPI_LAND);
        EXPECT_TRUE((mpi_operation_traits<std::logical_and<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::logical_and<>, T>::op()), MPI_LAND);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::logical_and<std::complex<int>>, T>::is_builtin));

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_or<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_or<T>, T>::op()), MPI_LOR);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_or<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_or<>, T>::op()), MPI_LOR);
        EXPECT_TRUE((mpi_operation_traits<std::logical_or<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::logical_or<>, T>::op()), MPI_LOR);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::logical_or<std::complex<int>>, T>::is_builtin));

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_xor<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_xor<T>, T>::op()), MPI_LXOR);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::logical_xor<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::logical_xor<>, T>::op()), MPI_LXOR);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::logical_xor<std::complex<int>>, T>::is_builtin));
    }

    if constexpr (kamping::mpi_type_traits<T>::is_integer || kamping::mpi_type_traits<T>::is_byte) {
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_and<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_and<T>, T>::op()), MPI_BAND);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_and<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_and<>, T>::op()), MPI_BAND);
        EXPECT_TRUE((mpi_operation_traits<std::bit_and<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::bit_and<>, T>::op()), MPI_BAND);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::bit_and<std::complex<int>>, T>::is_builtin));

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_or<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_or<T>, T>::op()), MPI_BOR);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_or<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_or<>, T>::op()), MPI_BOR);
        EXPECT_TRUE((mpi_operation_traits<std::bit_or<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<std::bit_or<>, T>::op()), MPI_BOR);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::bit_or<std::complex<int>>, T>::is_builtin));

        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_xor<T>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_xor<T>, T>::op()), MPI_BXOR);
        EXPECT_TRUE((mpi_operation_traits<kamping::ops::bit_xor<>, T>::is_builtin));
        EXPECT_EQ((mpi_operation_traits<kamping::ops::bit_xor<>, T>::op()), MPI_BXOR);
        EXPECT_FALSE((mpi_operation_traits<kamping::ops::bit_xor<std::complex<int>>, T>::is_builtin));
    }
}

TYPED_TEST(TypedOperationsTest, user_defined_operation_is_not_builtin) {
    using T = typename TestFixture::operation_type;
    auto op = [](auto a, auto b [[maybe_unused]]) {
        return a;
    };
    EXPECT_FALSE((kamping::internal::mpi_operation_traits<decltype(op), T>::is_builtin));
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
