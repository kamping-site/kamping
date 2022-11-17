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

#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(ReduceTest, reduce_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();
    if (comm.is_root()) {
        EXPECT_EQ(comm.root(), comm.size() - 1);
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to reduce
    for (size_t i = 0; i < comm.size(); ++i) {
        result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{}), root(i)).extract_recv_buffer();
        if (comm.rank() == i) {
            EXPECT_EQ(comm.root(), comm.size() - 1);
            EXPECT_EQ(result.size(), 2);
            EXPECT_EQ(result, expected_result);
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(ReduceTest, reduce_no_receive_buffer_bool) {
    Communicator comm;

    std::vector<kabool> input = {false, false};
    if (comm.rank() == 1 % comm.size()) {
        input[1] = true;
    }

    auto result = comm.reduce(send_buf(input), op(ops::logical_or<>{})).extract_recv_buffer();

    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), 2);
        std::vector<kabool> expected_result = {false, true};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_no_receive_buffer_kabool_custom_operation) {
    Communicator comm;

    std::vector<kabool> input = {false, false};
    if (comm.rank() == 1 % comm.size()) {
        input[1] = true;
    }

    // test that we can use a operation defined on bool even though wrap them as kabool
    auto my_or = [&](bool lhs, bool rhs) {
        return lhs || rhs;
    };
    auto result = comm.reduce(send_buf(input), op(my_or, ops::commutative)).extract_recv_buffer();

    if (comm.is_root()) {
        std::vector<kabool> expected_result = {false, true};
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_single_element_no_receive_buffer_kabool) {
    Communicator comm;

    kabool input = false;
    if (comm.rank() == 1 % comm.size()) {
        input = true;
    }

    auto result = comm.reduce(send_buf(input), op(ops::logical_or<>{})).extract_recv_buffer();

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 1);
        std::vector<kabool> expected_result = {true};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_single_element_initializer_list_bool_no_receive_buffer) {
    Communicator comm;

    bool input = false;
    if (comm.rank() == 1 % comm.size()) {
        input = true;
    }

    // reduce does not support single element bool when no recv_buf is specified, because the default would be
    // std::vector<bool>, which is not supported
    auto result = comm.reduce(send_buf({input}), op(ops::logical_or<>{})).extract_recv_buffer();

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 1);
        std::vector<kabool> expected_result = {true};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_single_element_explicit_receive_buffer_bool) {
    Communicator comm;

    bool               input = false;
    OwnContainer<bool> result;
    if (comm.rank() == 1 % comm.size()) {
        input = true;
    }

    comm.reduce(send_buf(input), recv_buf(result), op(ops::logical_or<>{}));

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 1);
        OwnContainer<bool> expected_result = {true};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.reduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf(result));

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};

    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    result = {};
    comm.root(comm.size() - 1);
    comm.reduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf(result));
    if (comm.is_root()) {
        EXPECT_EQ(comm.root(), comm.size() - 1);
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to reduce
    for (size_t i = 0; i < comm.size(); ++i) {
        result = {};
        comm.reduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf(result), root(i));
        if (comm.rank() == i) {
            EXPECT_EQ(comm.root(), comm.size() - 1);
            EXPECT_EQ(result.size(), 2);
            EXPECT_EQ(result, expected_result);
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(ReduceTest, reduce_with_receive_buffer_on_root) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    if (comm.is_root()) {
        std::vector<int> result;
        comm.reduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf(result));
        EXPECT_EQ(result.size(), 2);
        std::vector<int> expected_result = {
            (comm.size_signed() * (comm.size_signed() - 1)) / 2,
            comm.size_signed() * 42};
        EXPECT_EQ(result, expected_result);
    } else {
        auto result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_builtin_op_on_non_builtin_type) {
    Communicator comm;

    struct MyInt {
        MyInt() : _value(0) {}
        MyInt(int value) : _value(value) {}
        int _value;
        int operator+(MyInt const& rhs) const noexcept {
            return this->_value + rhs._value;
        }
        bool operator==(MyInt const& rhs) const {
            return this->_value == rhs._value;
        }
    };
    std::vector<MyInt> input = {comm.rank_signed(), 42};
    auto result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative)).extract_recv_buffer();
    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 2);
        std::vector<MyInt> expected_result = {
            (comm.size_signed() * (comm.size_signed() - 1)) / 2,
            comm.size_signed() * 42};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

int add_plus_42_function(int const& lhs, int const& rhs) {
    return lhs + rhs + 42;
}

TEST(ReduceTest, reduce_custom_operation_on_builtin_type) {
    Communicator comm;

    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };

    std::vector<int> input = {0, 17, 8};

    // use function ptr
    auto result = comm.reduce(send_buf(input), op(add_plus_42_function, kamping::ops::commutative)).extract_recv_buffer();

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // use lambda
    result = comm.reduce(send_buf(input), op(add_plus_42_lambda, kamping::ops::commutative)).extract_recv_buffer();

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // use lambda inline
    result = comm.reduce(
                     send_buf(input),
                     op([](auto const& lhs, auto const& rhs) { return lhs + rhs + 42; }, kamping::ops::commutative)
    )
                 .extract_recv_buffer();

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // use function object
    struct MySum42 {
        int operator()(int const& lhs, int const& rhs) {
            return lhs + rhs + 42;
        }
    };
    result = comm.reduce(send_buf(input), op(MySum42{}, kamping::ops::commutative)).extract_recv_buffer();

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.reduce(send_buf(input), op(get_right, kamping::ops::non_commutative)).extract_recv_buffer();

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 1);
        std::vector<int> expected_result = {comm.size_signed() - 1 + 17};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_custom_operation_on_custom_type) {
    Communicator comm;

    struct Aggregate {
        int  min;
        int  max;
        bool flag;
        int  sum;

        bool operator==(Aggregate const& rhs) const {
            return this->min == rhs.min && this->max == rhs.max && this->flag == rhs.flag && this->sum == rhs.sum;
        }
    };
    auto my_op = [](Aggregate const& lhs, Aggregate const& rhs) {
        Aggregate agg;
        agg.min  = std::min(lhs.min, rhs.min);
        agg.max  = std::max(lhs.max, rhs.max);
        agg.flag = lhs.flag || rhs.flag;
        agg.sum  = lhs.sum + rhs.sum;
        return agg;
    };

    Aggregate agg1 = {comm.rank_signed(), comm.rank_signed(), false, 1};
    if (comm.is_root()) {
        agg1.flag = true;
    }
    Aggregate              agg2  = {comm.rank_signed() + 42, comm.rank_signed() + 42, false, 1};
    std::vector<Aggregate> input = {agg1, agg2};

    Aggregate              agg1_expected   = {0, comm.size_signed() - 1, true, comm.size_signed()};
    Aggregate              agg2_expected   = {42, comm.size_signed() - 1 + 42, false, comm.size_signed()};
    std::vector<Aggregate> expected_result = {agg1_expected, agg2_expected};

    auto result = comm.reduce(send_buf(input), op(my_op, kamping::ops::commutative)).extract_recv_buffer();

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

// Death test do not work with MPI.
// TEST(ReduceTest, reduce_different_roots_on_different_processes) {
//     Communicator comm;
//     auto         value = comm.rank();
//
//     if (kassert::internal::assertion_enabled(assert::light_communication) && comm.size() > 1) {
//         EXPECT_KASSERT_FAILS(
//             comm.reduce(send_buf(value), op(kamping::ops::plus<>{}), root(comm.rank())),
//             "Root has to be the same on all ranks.");
//     }
// }
