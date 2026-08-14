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
#include <utility>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/mpi_ops.hpp"
#include "kamping/types/reduce_ops.hpp"

TEST(ScopedFunctorOpTest, test_local_reduction_stl_operation) {
    {
        kamping::types::ScopedFunctorOp<true, int, std::plus<>> op(std::plus<>{});
        std::array<int, 2>                                      a = {42, 69};
        std::array<int, 2>                                      b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get(), &commute);
        ASSERT_TRUE(commute);
    }
    {
        kamping::types::ScopedFunctorOp<false, int, std::plus<>> op(std::plus<>{});
        std::array<int, 2>                                       a = {42, 69};
        std::array<int, 2>                                       b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get(), &commute);
        ASSERT_FALSE(commute);
    }
}

TEST(ScopedFunctorOpTest, test_local_reduction_function_object) {
    struct MyOperation {
        int operator()(int const& a, int const& b) {
            return a + b;
        }
    };
    {
        kamping::types::ScopedFunctorOp<true, int, MyOperation> op(MyOperation{});
        std::array<int, 2>                                      a = {42, 69};
        std::array<int, 2>                                      b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get(), &commute);
        ASSERT_TRUE(commute);
    }
    {
        kamping::types::ScopedFunctorOp<false, int, MyOperation> op(MyOperation{});
        std::array<int, 2>                                       a = {42, 69};
        std::array<int, 2>                                       b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get(), &commute);
        ASSERT_FALSE(commute);
    }
}

// Copy-constructible but not assignable (mirrors std::map's/absl::flat_hash_map's value type,
// std::pair<const K, V>). ScopedFunctorOp::_execute() combines by destroying and
// reconstructing elements in place, not by assignment, so this must work end-to-end.
namespace {
struct NonAssignable {
    int value;
    NonAssignable(int v) : value(v) {}
    NonAssignable(NonAssignable const&)            = default;
    NonAssignable& operator=(NonAssignable const&) = delete;
    NonAssignable& operator=(NonAssignable&&)      = delete;
};
struct PickGreater {
    NonAssignable operator()(NonAssignable const& a, NonAssignable const& b) const {
        return a.value > b.value ? a : b;
    }
};
} // namespace

TEST(ScopedFunctorOpTest, test_local_reduction_non_assignable_value_type) {
    kamping::types::ScopedFunctorOp<true, NonAssignable, PickGreater> op(PickGreater{});
    std::array<NonAssignable, 2>                                      a = {NonAssignable{42}, NonAssignable{69}};
    std::array<NonAssignable, 2>                                      b = {NonAssignable{24}, NonAssignable{96}};
    // NonAssignable wraps a single int; reinterpret its bytes as MPI_INT for this local test.
    MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get());
    EXPECT_EQ(b[0].value, 42);
    EXPECT_EQ(b[1].value, 96);
}

// std::pair is never std::is_trivially_copyable_v -- its assignment operators aren't specified
// as defaulted/trivial even for two plain ints (see https://stackoverflow.com/q/58283694) --
// so ScopedFunctorOp must not require that trait, only destructibility and
// move-/copy-constructibility. This is the regression guard for that.
namespace {
struct PickGreaterFirst {
    std::pair<int, int> operator()(std::pair<int, int> const& a, std::pair<int, int> const& b) const {
        return a.first > b.first ? a : b;
    }
};
} // namespace

TEST(ScopedFunctorOpTest, test_local_reduction_ordinary_pair_value_type) {
    kamping::types::ScopedFunctorOp<true, std::pair<int, int>, PickGreaterFirst> op(PickGreaterFirst{});
    std::array<std::pair<int, int>, 2> a = {std::pair<int, int>{1, 10}, std::pair<int, int>{2, 20}};
    std::array<std::pair<int, int>, 2> b = {std::pair<int, int>{5, 50}, std::pair<int, int>{0, 0}};
    // std::pair<int,int> matches MPI_2INT's layout (two packed ints).
    MPI_Reduce_local(a.data(), b.data(), 2, MPI_2INT, op.get());
    EXPECT_EQ(b[0], (std::pair<int, int>{5, 50}));
    EXPECT_EQ(b[1], (std::pair<int, int>{2, 20}));
}

TEST(ScopedCallbackOpTest, test_local_reduction_with_wrapped_function_ptr) {
    kamping::types::ScopedCallbackOp<true>::callback_type op_ptr =
        [](void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
            int* invec_    = static_cast<int*>(invec);
            int* inoutvec_ = static_cast<int*>(inoutvec);
            std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, std::plus<>{});
        };
    {
        kamping::types::ScopedCallbackOp<true> op(op_ptr);
        std::array<int, 2>                     a = {42, 69};
        std::array<int, 2>                     b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get(), &commute);
        EXPECT_TRUE(commute);
    }
    {
        kamping::types::ScopedCallbackOp<false> op(op_ptr);
        std::array<int, 2>                      a = {42, 69};
        std::array<int, 2>                      b = {24, 96};
        MPI_Reduce_local(a.data(), b.data(), 2, MPI_INT, op.get());
        std::array<int, 2> expected_result = {42 + 24, 69 + 96};
        EXPECT_EQ(b, expected_result);

        int commute;
        MPI_Op_commutative(op.get(), &commute);
        EXPECT_FALSE(commute);
    }
}

template <typename T, typename Op, typename Commutative>
auto make_op(Op&& op, Commutative commutative) {
    return kamping::internal::ReduceOperation<T, Op, Commutative>(std::move(op), commutative);
}

void my_plus(void* invec, void* inoutvec, int* len, MPI_Datatype* type) {
    KAMPING_ASSERT(*type == MPI_INT);
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
