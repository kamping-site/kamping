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

#include "../test_assertions.hpp"

#include <gmock/gmock-matchers.h>
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

    auto result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{}));

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{}));
    if (comm.is_root()) {
        EXPECT_EQ(comm.root(), comm.size() - 1);
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to reduce
    for (size_t i = 0; i < comm.size(); ++i) {
        result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{}), root(i));
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

    auto result = comm.reduce(send_buf(input), op(ops::logical_or<>{}));

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
    auto result = comm.reduce(send_buf(input), op(my_or, ops::commutative));

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

    auto result = comm.reduce(send_buf(input), op(ops::logical_or<>{}));

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
    auto result = comm.reduce(send_buf({input}), op(ops::logical_or<>{}));

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

    comm.reduce(send_buf(input), recv_buf<kamping::BufferResizePolicy::resize_to_fit>(result), op(ops::logical_or<>{}));

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 1);
        OwnContainer<bool> expected_result = {true};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_single_element_explicit_receive_buffer_bool_no_resize) {
    Communicator comm;

    bool               input = false;
    OwnContainer<bool> result(3, false);
    if (comm.rank() == 1 % comm.size()) {
        input = true;
    }

    comm.reduce(
        send_buf(input),
        recv_buf<kamping::BufferResizePolicy::no_resize>(result),
        send_recv_count(1),
        op(ops::logical_or<>{})
    );

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 3);
        OwnContainer<bool> expected_result = {true, false, false};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 3);
        OwnContainer<bool> expected_result = {false, false, false};
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ReduceTest, reduce_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.reduce(
        send_buf(input),
        op(kamping::ops::plus<>{}),
        recv_buf<kamping::BufferResizePolicy::resize_to_fit>(result)
    );

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
    comm.reduce(
        send_buf(input),
        op(kamping::ops::plus<>{}),
        recv_buf<kamping::BufferResizePolicy::resize_to_fit>(result)
    );
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
        comm.reduce(
            send_buf(input),
            op(kamping::ops::plus<>{}),
            recv_buf<kamping::BufferResizePolicy::resize_to_fit>(result),
            root(i)
        );
        if (comm.rank() == i) {
            EXPECT_EQ(comm.root(), comm.size() - 1);
            EXPECT_EQ(result.size(), 2);
            EXPECT_EQ(result, expected_result);
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(ReduceTest, reduce_with_receive_buffer_no_resize_and_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input  = {1, 2, 3, 4};
    std::vector<int> result = {42, 42};

    comm.reduce(
        send_buf(input),
        op(kamping::ops::plus<>{}),
        recv_buf<kamping::BufferResizePolicy::no_resize>(result),
        send_recv_count(1)
    );

    if (comm.rank() == comm.root()) {
        EXPECT_THAT(result, ElementsAre(comm.size(), 42));
    } else {
        EXPECT_THAT(result, ElementsAre(42, 42));
    }
}

TEST(ReduceTest, reduce_with_receive_buffer_resize_to_fit_and_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input  = {1, 2, 3, 4};
    std::vector<int> result = {42, 42};

    comm.reduce(
        send_buf(input),
        op(kamping::ops::plus<>{}),
        recv_buf<kamping::BufferResizePolicy::resize_to_fit>(result),
        send_recv_count(1)
    );

    if (comm.rank() == comm.root()) {
        EXPECT_THAT(result, ElementsAre(comm.size()));
    } else {
        // do not touch the buffer on non root rank
        EXPECT_THAT(result, ElementsAre(42, 42));
    }
}

TEST(ReduceTest, reduce_with_receive_buffer_grow_only_and_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input  = {1, 2, 3, 4};
    std::vector<int> result = {42, 42};

    comm.reduce(
        send_buf(input),
        op(kamping::ops::plus<>{}),
        recv_buf<kamping::BufferResizePolicy::grow_only>(result),
        send_recv_count(1)
    );

    // not resized to 1 because big enough
    if (comm.rank() == comm.root()) {
        EXPECT_THAT(result, ElementsAre(comm.size(), 42));
    } else {
        EXPECT_THAT(result, ElementsAre(42, 42));
    }
}

TEST(ReduceTest, reduce_with_receive_buffer_on_root) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    if (comm.is_root()) {
        std::vector<int> result;
        comm.reduce(
            send_buf(input),
            op(kamping::ops::plus<>{}),
            recv_buf<kamping::BufferResizePolicy::resize_to_fit>(result)
        );
        EXPECT_EQ(result.size(), 2);
        std::vector<int> expected_result = {
            (comm.size_signed() * (comm.size_signed() - 1)) / 2,
            comm.size_signed() * 42};
        EXPECT_EQ(result, expected_result);
    } else {
        auto result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{}));
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_builtin_op_on_non_builtin_type) {
    Communicator comm;

    struct MyInt {
        MyInt() noexcept : _value(0) {}
        MyInt(int value) noexcept : _value(value) {}
        int _value;
        int operator+(MyInt const& rhs) const noexcept {
            return this->_value + rhs._value;
        }
        bool operator==(MyInt const& rhs) const noexcept {
            return this->_value == rhs._value;
        }
    };
    std::vector<MyInt> input  = {comm.rank_signed(), 42};
    auto               result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative));
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
    auto result = comm.reduce(send_buf(input), op(add_plus_42_function, kamping::ops::commutative));

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
    result = comm.reduce(send_buf(input), op(add_plus_42_lambda, kamping::ops::commutative));

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
    );

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
    result = comm.reduce(send_buf(input), op(MySum42{}, kamping::ops::commutative));

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

TEST(ReduceTest, reduce_builtin_native_operation) {
    Communicator comm;

    std::vector<int> input = {1, 2, 3};

    auto result = comm.reduce(send_buf(input), op(MPI_SUM));

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {comm.size_signed() * 1, comm.size_signed() * 2, comm.size_signed() * 3};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST(ReduceTest, reduce_builtin_native_operation_with_incompatible_type) {
    struct MyInt {
        MyInt() noexcept : _value(0) {}
        MyInt(int value) noexcept : _value(value) {}
        int _value;
        int operator+(MyInt const& rhs) const noexcept {
            return this->_value + rhs._value;
        }
        bool operator==(MyInt const& rhs) const noexcept {
            return this->_value == rhs._value;
        }
    };
    Communicator comm;

    std::vector<MyInt> input = {1, 2, 3};

    EXPECT_KASSERT_FAILS(
        auto result = comm.reduce(send_buf(input), op(MPI_SUM)),
        "The provided builtin operation is not compatible with datatype T."
    )
}
#endif

void select_left_op_func(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
    EXPECT_EQ(*datatype, MPI_INT);
    int* invec_    = static_cast<int*>(invec);
    int* inoutvec_ = static_cast<int*>(inoutvec);
    std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, [](int const& left, int const&) { return left; });
}

TEST(ReduceTest, reduce_builtin_handmade_native_operation) {
    Communicator comm;

    MPI_Op select_left_op;
    MPI_Op_create(&select_left_op_func, false, &select_left_op);

    std::vector<int> input = {1 + comm.rank_signed(), 2 + comm.rank_signed(), 3 + comm.rank_signed()};

    auto result = comm.reduce(send_buf(input), op(select_left_op));

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {1, 2, 3};
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
    MPI_Op_free(&select_left_op);
}

TEST(ReduceTest, reduce_custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.reduce(send_buf(input), op(get_right, kamping::ops::non_commutative));

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

    auto result = comm.reduce(send_buf(input), op(my_op, kamping::ops::commutative));

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), 2);
        EXPECT_EQ(result, expected_result);
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(ReduceTest, reduce_default_container_type) {
    Communicator<OwnContainer> comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    OwnContainer<int> result = comm.reduce(send_buf(input), op(kamping::ops::plus<>{}));
}

TEST(ReduceTest, reduce_custom_operation_on_custom_mpi_type) {
    Communicator comm;
    int const    dont_care = -1;

    struct Aggregate {
        int min;
        int padding = dont_care;
        int max;

        bool operator==(Aggregate const& rhs) const {
            return this->min == rhs.min && this->max == rhs.max;
        }
    };
    MPI_Datatype int_padding_int = MPI_INT_padding_MPI_INT();
    auto         my_op           = [](Aggregate const& lhs, Aggregate const& rhs) {
        Aggregate agg;
        agg.min = std::min(lhs.min, rhs.min);
        agg.max = std::max(lhs.max, rhs.max);
        return agg;
    };

    Aggregate              agg1  = {comm.rank_signed(), dont_care, comm.rank_signed()};
    Aggregate              agg2  = {comm.rank_signed() + 42, dont_care, comm.rank_signed() + 42};
    std::vector<Aggregate> input = {agg1, agg2};

    Aggregate                    agg1_expected   = {0, dont_care, comm.size_signed() - 1};
    Aggregate                    agg2_expected   = {42, dont_care, comm.size_signed() - 1 + 42};
    std::vector<Aggregate> const expected_result = {agg1_expected, agg2_expected};
    std::vector<Aggregate>       recv_buffer(2);
    int const                    root_rank = 0;

    MPI_Type_commit(&int_padding_int);
    comm.reduce(
        send_buf(input),
        send_recv_count(2),
        send_recv_type(int_padding_int),
        op(my_op, kamping::ops::commutative),
        root(root_rank),
        recv_buf<no_resize>(recv_buffer)
    );
    MPI_Type_free(&int_padding_int);

    if (comm.is_root(root_rank)) {
        EXPECT_EQ(recv_buffer, expected_result);
    }
}

void sum_for_int_padding_padding_type(void* in_buf, void* inout_buf, int* len, MPI_Datatype*) {
    kamping::Communicator<> comm;
    int*                    in_buffer    = reinterpret_cast<int*>(in_buf);
    int*                    inout_buffer = reinterpret_cast<int*>(inout_buf);
    for (size_t i = 0; i < static_cast<size_t>(*len); ++i) {
        inout_buffer[3 * i] = in_buffer[3 * i] + inout_buffer[3 * i];
    }
}

TEST(ReduceTest, reduce_custom_operation_on_custom_mpi_without_matching_cpp_type) {
    Communicator comm;
    int const    dont_care = -1;

    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input = {comm.rank_signed(), dont_care, dont_care, comm.rank_signed() + 42, dont_care, dont_care};

    int const              sum_of_ranks = comm.size_signed() * (comm.size_signed() - 1) / 2;
    std::vector<int> const expected_result =
        {sum_of_ranks, dont_care, dont_care, sum_of_ranks + comm.size_signed() * 42, dont_care, dont_care};
    std::vector<int> recv_buffer(6, dont_care);
    int const        root_rank = 0;

    MPI_Op user_defined_op;
    MPI_Op_create(sum_for_int_padding_padding_type, 1, &user_defined_op);
    MPI_Type_commit(&int_padding_padding);
    comm.reduce(
        send_buf(input),
        send_recv_count(2),
        send_recv_type(int_padding_padding),
        op(user_defined_op),
        root(root_rank),
        recv_buf<no_resize>(recv_buffer)
    );
    MPI_Type_free(&int_padding_padding);
    MPI_Op_free(&user_defined_op);

    if (comm.is_root(root_rank)) {
        EXPECT_EQ(recv_buffer, expected_result);
    }
}
TEST(ReduceTest, send_recv_type_is_out_parameter) {
    Communicator           comm;
    std::vector<int> const data{1};
    MPI_Datatype           send_type;
    int const              root_rank = 0;
    auto                   recv_buf =
        comm.reduce(send_buf(data), send_recv_type_out(send_type), op(kamping::ops::plus<>{}), root(root_rank));

    EXPECT_EQ(send_type, MPI_INT);
    if (comm.is_root(root_rank)) {
        EXPECT_EQ(recv_buf.size(), 1);
        EXPECT_EQ(recv_buf.front(), comm.size());
    } else {
        EXPECT_EQ(recv_buf.size(), 0);
    }
}

TEST(ReduceTest, send_type_part_of_result_object) {
    Communicator           comm;
    std::vector<int> const data{1};
    int const              root_rank = 0;
    auto result = comm.reduce(send_buf(data), send_recv_type_out(), op(kamping::ops::plus<>{}), root(root_rank));

    EXPECT_EQ(result.extract_send_recv_type(), MPI_INT);
    auto recv_buf = result.extract_recv_buffer();
    if (comm.is_root(root_rank)) {
        EXPECT_EQ(recv_buf.size(), 1);
        EXPECT_EQ(recv_buf.front(), comm.size());
    } else {
        EXPECT_EQ(recv_buf.size(), 0);
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

TEST(ReduceTest, reduce_single) {
    Communicator       comm;
    int                input  = comm.rank_signed();
    std::optional<int> result = comm.reduce_single(send_buf(input), op(kamping::ops::plus<>{}));

    if (comm.is_root()) {
        int const expected_result = (comm.size_signed() * (comm.size_signed() - 1)) / 2;
        EXPECT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), expected_result);
    } else {
        EXPECT_EQ(result, std::nullopt);
    }
}

TEST(ReduceTest, reduce_single_with_temporary) {
    Communicator       comm;
    std::optional<int> result = comm.reduce_single(send_buf(comm.rank_signed()), op(kamping::ops::plus<>{}));

    if (comm.is_root()) {
        int const expected_result = (comm.size_signed() * (comm.size_signed() - 1)) / 2;
        EXPECT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), expected_result);
    } else {
        EXPECT_EQ(result, std::nullopt);
    }
}

TEST(ReduceTest, reduce_single_with_root_param) {
    Communicator comm;

    int                input  = comm.rank_signed();
    int const          root   = comm.size_signed() - 1;
    std::optional<int> result = comm.reduce_single(send_buf(input), kamping::root(root), op(kamping::ops::plus<>{}));

    if (comm.is_root(root)) {
        int const expected_result = (comm.size_signed() * (comm.size_signed() - 1)) / 2;
        EXPECT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), expected_result);
    } else {
        EXPECT_EQ(result, std::nullopt);
    }
}

TEST(ReduceTest, structured_bindings_explicit_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t>       values{comm.rank(), comm.rank()};
    std::vector<std::uint64_t> const expected_recv_buffer_on_root(2, comm.size() * (comm.size() - 1) / 2);
    std::vector<std::uint64_t>       recv_buffer;
    int const                        root        = comm.size_signed() - 1;
    auto const [send_recv_type, send_recv_count] = comm.reduce(
        send_recv_type_out(),
        send_recv_count_out(),
        send_buf(values),
        recv_buf<resize_to_fit>(recv_buffer),
        op(kamping::ops::plus<>{}),
        kamping::root(root)
    );

    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
    if (comm.is_root(root)) {
        EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
    } else {
        EXPECT_TRUE(recv_buffer.empty());
    }
}

TEST(ReduceTest, structured_bindings_explicit_owning_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t>       values{comm.rank(), comm.rank()};
    std::vector<std::uint64_t> const expected_recv_buffer_on_root(2, comm.size() * (comm.size() - 1) / 2);
    auto const [send_recv_type, send_recv_count, recv_buffer] = comm.reduce(
        send_recv_type_out(),
        send_recv_count_out(),
        send_buf(values),
        recv_buf(alloc_new<std::vector<std::uint64_t>>),
        op(kamping::ops::plus<>{})
    );

    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
    if (comm.is_root()) {
        EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
    } else {
        EXPECT_TRUE(recv_buffer.empty());
    }
}

TEST(ReduceTest, structured_bindings_implicit_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t>       values{comm.rank(), comm.rank()};
    std::vector<std::uint64_t> const expected_recv_buffer_on_root(2, comm.size() * (comm.size() - 1) / 2);
    {
        std::vector<std::uint64_t> tmp(2);
        auto const [recv_buffer, send_recv_type, send_recv_count] =
            comm.reduce(send_recv_type_out(), send_recv_count_out(), send_buf(values), op(kamping::ops::plus<>{}));

        EXPECT_EQ(send_recv_count, 2);
        EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
        if (comm.is_root()) {
            EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
        } else {
            EXPECT_TRUE(recv_buffer.empty());
        }
    }
    {
        // non-owning send_recv_type out buffer
        std::vector<std::uint64_t> tmp(2);
        MPI_Datatype               send_recv_type;
        auto const [recv_buffer, send_recv_count] = comm.reduce(
            send_recv_type_out(send_recv_type),
            send_recv_count_out(),
            send_buf(values),
            op(kamping::ops::plus<>{})
        );

        EXPECT_EQ(send_recv_count, 2);
        EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
        if (comm.is_root()) {
            EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
        } else {
            EXPECT_TRUE(recv_buffer.empty());
        }
    }
}
