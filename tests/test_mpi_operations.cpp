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
TYPED_TEST_SUITE(TypedOperationsTest, MyTypes);

TEST(TypeGroupsTest, test_type_groups) {
    EXPECT_TRUE(kamping::is_mpi_integer<int32_t>());
    EXPECT_FALSE(kamping::is_mpi_float<int32_t>());
    EXPECT_FALSE(kamping::is_mpi_logical<int32_t>());
    EXPECT_FALSE(kamping::is_mpi_complex<int32_t>());
    EXPECT_FALSE(kamping::is_mpi_byte<int32_t>());

    EXPECT_TRUE(kamping::is_mpi_integer<u_int32_t>());
    EXPECT_FALSE(kamping::is_mpi_float<u_int32_t>());
    EXPECT_FALSE(kamping::is_mpi_logical<u_int32_t>());
    EXPECT_FALSE(kamping::is_mpi_complex<u_int32_t>());
    EXPECT_FALSE(kamping::is_mpi_byte<u_int32_t>());

    EXPECT_TRUE(kamping::is_mpi_integer<int64_t>());
    EXPECT_FALSE(kamping::is_mpi_float<int64_t>());
    EXPECT_FALSE(kamping::is_mpi_logical<int64_t>());
    EXPECT_FALSE(kamping::is_mpi_complex<int64_t>());
    EXPECT_FALSE(kamping::is_mpi_byte<int64_t>());

    EXPECT_TRUE(kamping::is_mpi_integer<u_int64_t>());
    EXPECT_FALSE(kamping::is_mpi_float<u_int64_t>());
    EXPECT_FALSE(kamping::is_mpi_logical<u_int64_t>());
    EXPECT_FALSE(kamping::is_mpi_complex<u_int64_t>());
    EXPECT_FALSE(kamping::is_mpi_byte<u_int64_t>());

    EXPECT_FALSE(kamping::is_mpi_integer<float>());
    EXPECT_TRUE(kamping::is_mpi_float<float>());
    EXPECT_FALSE(kamping::is_mpi_logical<float>());
    EXPECT_FALSE(kamping::is_mpi_complex<float>());
    EXPECT_FALSE(kamping::is_mpi_byte<float>());

    EXPECT_FALSE(kamping::is_mpi_integer<double>());
    EXPECT_TRUE(kamping::is_mpi_float<double>());
    EXPECT_FALSE(kamping::is_mpi_logical<double>());
    EXPECT_FALSE(kamping::is_mpi_complex<double>());
    EXPECT_FALSE(kamping::is_mpi_byte<double>());

    EXPECT_FALSE(kamping::is_mpi_integer<std::complex<double>>());
    EXPECT_FALSE(kamping::is_mpi_float<std::complex<double>>());
    EXPECT_FALSE(kamping::is_mpi_logical<std::complex<double>>());
    EXPECT_TRUE(kamping::is_mpi_complex<std::complex<double>>());
    EXPECT_FALSE(kamping::is_mpi_byte<std::complex<double>>());

    EXPECT_FALSE(kamping::is_mpi_integer<DummyType>());
    EXPECT_FALSE(kamping::is_mpi_float<DummyType>());
    EXPECT_FALSE(kamping::is_mpi_logical<DummyType>());
    EXPECT_FALSE(kamping::is_mpi_complex<DummyType>());
    EXPECT_FALSE(kamping::is_mpi_byte<DummyType>());
}


TYPED_TEST(TypedOperationsTest, test_builtin_operations) {
    using namespace kamping::internal;
    using T = typename TestFixture::operation_type;

    if constexpr (kamping::is_mpi_integer<T>() || kamping::is_mpi_float<T>()) {
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

    if constexpr (kamping::is_mpi_integer<T>() || kamping::is_mpi_float<T>() || kamping::is_mpi_complex<T>()) {
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

    if constexpr (kamping::is_mpi_integer<T>() || kamping::is_mpi_logical<T>()) {
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

    if constexpr (kamping::is_mpi_integer<T>() || kamping::is_mpi_byte<T>()) {
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
    auto op = [](auto a, auto b) {
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

