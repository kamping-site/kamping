#include <cstdint>
#include <gtest/gtest.h>
#include <sys/types.h>

#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_ops.hpp"

template <typename T>
class OperationsTest : public ::testing::Test {
public:
    using operation_type = T;
};

using MyTypes = ::testing::Types<int32_t, u_int32_t, int64_t, u_int64_t, float, double>;
TYPED_TEST_SUITE(OperationsTest, MyTypes);

TYPED_TEST(OperationsTest, test_builtin_operations) {
    using namespace kamping::internal;
    using T = typename TestFixture::operation_type;

    if constexpr (kamping::is_mpi_integer<T>() || kamping::is_mpi_float<T>()) {
        // static_assert(kamping::is_mpi_integer<T>(), "is not mpi");
        // static_assert(!kamping::is_mpi_float<T>(), "is not mpi");
        EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::plus<T>, T>::value));
        EXPECT_TRUE((is_builtin_mpi_op<std::plus<T>, T>::value));
        EXPECT_TRUE((is_builtin_mpi_op<std::plus<>, T>::value));
    }

    EXPECT_EQ((is_builtin_mpi_op<kamping::ops::plus<T>, T>::op()), MPI_SUM);

    EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::multiplies<T>, T>::value));
    EXPECT_EQ((is_builtin_mpi_op<kamping::ops::multiplies<T>, T>::op()), MPI_PROD);

    EXPECT_TRUE((is_builtin_mpi_op<kamping::ops::multiplies<T>, T>::value));
    EXPECT_EQ((is_builtin_mpi_op<kamping::ops::multiplies<T>, T>::op()), MPI_PROD);
}
