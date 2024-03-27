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

#include <array>
#include <type_traits>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/mpi_ops.hpp"

TEST(UserOperationWrapperTest, test_local_reduction_stl_operation) {
    {
        kamping::internal::UserOperationWrapper<true, int, std::plus<>> op(std::plus<>{});
        std::array<int, 2>                                              a = {42, 69};
        std::array<int, 2>                                              b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get_mpi_op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get_mpi_op(), &commute);
        ASSERT_TRUE(commute);
    }
    {
        kamping::internal::UserOperationWrapper<false, int, std::plus<>> op(std::plus<>{});
        std::array<int, 2>                                               a = {42, 69};
        std::array<int, 2>                                               b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get_mpi_op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get_mpi_op(), &commute);
        ASSERT_FALSE(commute);
    }
}

TEST(UserOperationWrapperTest, test_local_reduction_function_object) {
    struct MyOperation {
        int operator()(int const& a, int const& b) {
            return a + b;
        }
    };
    {
        kamping::internal::UserOperationWrapper<true, int, MyOperation> op(MyOperation{});
        std::array<int, 2>                                              a = {42, 69};
        std::array<int, 2>                                              b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get_mpi_op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get_mpi_op(), &commute);
        ASSERT_TRUE(commute);
    }
    {
        kamping::internal::UserOperationWrapper<false, int, MyOperation> op(MyOperation{});
        std::array<int, 2>                                               a = {42, 69};
        std::array<int, 2>                                               b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get_mpi_op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get_mpi_op(), &commute);
        ASSERT_FALSE(commute);
    }
}

TEST(UserOperationPtrWrapper, test_local_reduction_with_wrapped_function_ptr) {
    kamping::internal::mpi_custom_operation_type op_ptr =
        [](void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
            int* invec_    = static_cast<int*>(invec);
            int* inoutvec_ = static_cast<int*>(inoutvec);
            std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, std::plus<>{});
        };
    {
        kamping::internal::UserOperationPtrWrapper<true> op(op_ptr);
        std::array<int, 2>                               a = {42, 69};
        std::array<int, 2>                               b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get_mpi_op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get_mpi_op(), &commute);
        EXPECT_TRUE(commute);
    }
    {
        kamping::internal::UserOperationPtrWrapper<false> op(op_ptr);
        std::array<int, 2>                                a = {42, 69};
        std::array<int, 2>                                b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get_mpi_op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get_mpi_op(), &commute);
        EXPECT_FALSE(commute);
    }
}

template <typename T, typename Op, typename Commutative>
auto make_op(Op&& op, Commutative commutative) {
    return kamping::internal::ReduceOperation<T, Op, Commutative>(std::move(op), commutative);
}

void my_plus(void* invec, void* inoutvec, int* len, MPI_Datatype* type) {
    KASSERT(*type == MPI_INT);
    int* invec_    = static_cast<int*>(invec);
    int* inoutvec_ = static_cast<int*>(inoutvec);
    std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, std::plus<>{});
}

TEST(ReduceOperationTest, test_dispatch_for_builtin_function_object_and_lambda) {
    struct WrappedInt {
        int        value;
        WrappedInt operator+(WrappedInt const& a) const noexcept {
            return {this->value + a.value};
        }
        bool operator==(WrappedInt const& a) const noexcept {
            return this->value == a.value;
        }
    };
    // builtin operation
    {
        auto op = make_op<int>(std::plus<>{}, kamping::ops::internal::undefined_commutative_tag{});
        EXPECT_EQ(op.op(), MPI_SUM);
        EXPECT_EQ(op(3, 4), 7);
        EXPECT_TRUE(decltype(op)::is_builtin);

        std::array<int, 2> a = {42, 69};
        std::array<int, 2> b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        EXPECT_TRUE(decltype(op)::commutative);
        int commute;
        MPI_Op_commutative(op.op(), &commute);
        EXPECT_TRUE(commute);
    }
    // builtin operation on non-builtin type commutative
    {
        auto op = make_op<WrappedInt>(std::plus<>{}, kamping::ops::commutative);
        EXPECT_NE(op.op(), MPI_SUM);
        EXPECT_EQ(op(WrappedInt{3}, WrappedInt{4}), WrappedInt{7});
        EXPECT_FALSE(decltype(op)::is_builtin);

        std::array<int, 2> a = {42, 69};
        std::array<int, 2> b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        EXPECT_TRUE(decltype(op)::commutative);
        int commute;
        MPI_Op_commutative(op.op(), &commute);
        EXPECT_TRUE(commute);
    }
    // builtin operation on non-builtin type non-commutative
    {
        auto op = make_op<WrappedInt>(std::plus<>{}, kamping::ops::non_commutative);
        EXPECT_NE(op.op(), MPI_SUM);
        EXPECT_EQ(op(WrappedInt{3}, WrappedInt{4}), WrappedInt{7});
        EXPECT_FALSE(decltype(op)::is_builtin);

        std::array<int, 2> a = {42, 69};
        std::array<int, 2> b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        EXPECT_FALSE(decltype(op)::commutative);
        int commute;
        MPI_Op_commutative(op.op(), &commute);
        EXPECT_FALSE(commute);
    }
    // builtin native operation
    {
        auto op = make_op<int>(MPI_SUM, kamping::ops::internal::undefined_commutative_tag{});
        EXPECT_EQ(op.op(), MPI_SUM);
        EXPECT_EQ(op(3, 4), 7);
        EXPECT_FALSE(decltype(op)::is_builtin);
    }
    // custom native operation
    {
        MPI_Op native_op;
        MPI_Op_create(my_plus, true, &native_op);
        auto op =
            kamping::internal::ReduceOperation<int, MPI_Op, kamping::ops::internal::undefined_commutative_tag>(native_op
            );
        EXPECT_EQ(op.op(), native_op);
        EXPECT_EQ(op(3, 4), 7);
        EXPECT_FALSE(decltype(op)::is_builtin);
    }
    // lambda on builtin type commutative
    {
        auto op = make_op<int>([](auto a, auto b) { return a + b; }, kamping::ops::commutative);
        EXPECT_NE(op.op(), MPI_SUM);
        EXPECT_EQ(op(3, 4), 7);
        EXPECT_FALSE(decltype(op)::is_builtin);

        std::array<int, 2> a = {42, 69};
        std::array<int, 2> b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        EXPECT_TRUE(decltype(op)::commutative);
        int commute;
        MPI_Op_commutative(op.op(), &commute);
        EXPECT_TRUE(commute);
    }
    // lambda on builtin type non-commutative
    {
        auto op = make_op<int>([](auto a, auto b) { return a + b; }, kamping::ops::non_commutative);
        EXPECT_NE(op.op(), MPI_SUM);
        EXPECT_EQ(op(3, 4), 7);
        EXPECT_FALSE(decltype(op)::is_builtin);

        std::array<int, 2> a = {42, 69};
        std::array<int, 2> b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        EXPECT_FALSE(decltype(op)::commutative);
        int commute;
        MPI_Op_commutative(op.op(), &commute);
        EXPECT_FALSE(commute);
    }
    // lambda on custom type commutative
    {
        auto op = make_op<WrappedInt>([](auto a, auto b) { return a + b; }, kamping::ops::commutative);
        EXPECT_NE(op.op(), MPI_SUM);
        EXPECT_EQ(op(WrappedInt{3}, WrappedInt{4}), WrappedInt{7});
        EXPECT_FALSE(decltype(op)::is_builtin);

        std::array<int, 2> a = {42, 69};
        std::array<int, 2> b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        EXPECT_TRUE(decltype(op)::commutative);
        int commute;
        MPI_Op_commutative(op.op(), &commute);
        EXPECT_TRUE(commute);
    }
    // lambda on custom type non-commutative
    {
        auto op = make_op<WrappedInt>([](auto a, auto b) { return a + b; }, kamping::ops::non_commutative);
        EXPECT_NE(op.op(), MPI_SUM);
        EXPECT_EQ(op(WrappedInt{3}, WrappedInt{4}), WrappedInt{7});
        EXPECT_FALSE(decltype(op)::is_builtin);

        std::array<int, 2> a = {42, 69};
        std::array<int, 2> b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.op());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        EXPECT_FALSE(decltype(op)::commutative);
        int commute;
        MPI_Op_commutative(op.op(), &commute);
        EXPECT_FALSE(commute);
    }
}
