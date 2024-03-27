// This file is part of KaMPIng.
//
// Copyright 2022-2023 The KaMPIng Authors
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(AllreduceTest, allreduce_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto result = comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_with_receive_buffer_resize_too_big) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result(10, -1);

    comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_with_receive_buffer_no_resize_and_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input  = {1, 2, 3, 4};
    std::vector<int> result = {42, 42};

    comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<no_resize>(result), send_recv_count(1));
    EXPECT_THAT(result, ElementsAre(comm.size(), 42));
}

TEST(AllreduceTest, allreduce_with_receive_buffer_grow_only_and_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input  = {1, 2, 3, 4};
    std::vector<int> result = {42, 42};

    comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<grow_only>(result), send_recv_count(1));
    EXPECT_THAT(result, ElementsAre(comm.size(), 42));
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
TEST(AllreduceTest, allreduce_with_receive_buffer_no_resize_too_small) {
    Communicator comm;

    std::vector<int> input = {1, 2, 3, 4};
    std::vector<int> result;

    EXPECT_KASSERT_FAILS(
        {
            comm.allreduce(
                send_buf(input),
                op(kamping::ops::plus<>{}),
                recv_buf<kamping::BufferResizePolicy::no_resize>(result),
                send_recv_count(1)
            );
        },
        ""
    );
}
#endif

TEST(AllreduceTest, allreduce_builtin_op_on_non_builtin_type) {
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
    std::vector<MyInt> input = {comm.rank_signed(), 42};

    auto result = comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative));
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

int add_plus_42_function(int const& lhs, int const& rhs) {
    return lhs + rhs + 42;
}

TEST(AllreduceTest, allreduce_custom_operation_on_builtin_type) {
    Communicator comm;

    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };

    std::vector<int> input = {0, 17, 8};

    { // use function ptr
        auto result = comm.allreduce(send_buf(input), op(add_plus_42_function, kamping::ops::commutative));

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use lambda
        auto result = comm.allreduce(send_buf(input), op(add_plus_42_lambda, kamping::ops::commutative));

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use lambda inline
        auto result = comm.allreduce(
            send_buf(input),
            op([](auto const& lhs, auto const& rhs) { return lhs + rhs + 42; }, kamping::ops::commutative)
        );

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use function object
        struct MySum42 {
            int operator()(int const& lhs, int const& rhs) {
                return lhs + rhs + 42;
            }
        };
        auto result = comm.allreduce(send_buf(input), op(MySum42{}, kamping::ops::commutative));

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    }
}

TEST(AllreduceTest, allreduce_custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.allreduce(send_buf(input), op(get_right, kamping::ops::non_commutative));

    EXPECT_EQ(result.size(), 1);
    std::vector<int> expected_result = {comm.size_signed() - 1 + 17};
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_custom_operation_on_custom_type) {
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

    Aggregate agg1               = {comm.rank_signed(), comm.rank_signed(), false, 1};
    agg1.flag                    = true;
    Aggregate              agg2  = {comm.rank_signed() + 42, comm.rank_signed() + 42, false, 1};
    std::vector<Aggregate> input = {agg1, agg2};

    Aggregate              agg1_expected   = {0, comm.size_signed() - 1, true, comm.size_signed()};
    Aggregate              agg2_expected   = {42, comm.size_signed() - 1 + 42, false, comm.size_signed()};
    std::vector<Aggregate> expected_result = {agg1_expected, agg2_expected};

    auto result = comm.allreduce(send_buf(input), op(my_op, kamping::ops::commutative));

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_custom_operation_on_custom_mpi_type) {
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

    Aggregate              agg1_expected   = {0, dont_care, comm.size_signed() - 1};
    Aggregate              agg2_expected   = {42, dont_care, comm.size_signed() - 1 + 42};
    std::vector<Aggregate> expected_result = {agg1_expected, agg2_expected};
    std::vector<Aggregate> recv_buffer(2);

    MPI_Type_commit(&int_padding_int);
    comm.allreduce(
        send_buf(input),
        send_recv_count(2),
        send_recv_type(int_padding_int),
        op(my_op, kamping::ops::commutative),
        recv_buf<no_resize>(recv_buffer)
    );
    MPI_Type_free(&int_padding_int);

    EXPECT_EQ(recv_buffer, expected_result);
}

void sum_for_int_padding_padding_type(void* in_buf, void* inout_buf, int* len, MPI_Datatype*) {
    kamping::Communicator<> comm;
    int*                    in_buffer    = reinterpret_cast<int*>(in_buf);
    int*                    inout_buffer = reinterpret_cast<int*>(inout_buf);
    for (size_t i = 0; i < static_cast<size_t>(*len); ++i) {
        inout_buffer[3 * i] = in_buffer[3 * i] + inout_buffer[3 * i];
    }
}

TEST(AllreduceTest, allreduce_custom_operation_on_custom_mpi_without_matching_cpp_type) {
    Communicator comm;
    int const    dont_care = -1;

    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input = {comm.rank_signed(), dont_care, dont_care, comm.rank_signed() + 42, dont_care, dont_care};

    int const        sum_of_ranks = comm.size_signed() * (comm.size_signed() - 1) / 2;
    std::vector<int> expected_result =
        {sum_of_ranks, dont_care, dont_care, sum_of_ranks + comm.size_signed() * 42, dont_care, dont_care};
    std::vector<int> recv_buffer(6, dont_care);

    MPI_Op user_defined_op;
    MPI_Op_create(sum_for_int_padding_padding_type, 1, &user_defined_op);
    MPI_Type_commit(&int_padding_padding);
    comm.allreduce(
        send_buf(input),
        send_recv_count(2),
        send_recv_type(int_padding_padding),
        op(user_defined_op),
        recv_buf<no_resize>(recv_buffer)
    );
    MPI_Type_free(&int_padding_padding);
    MPI_Op_free(&user_defined_op);

    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AllreduceTest, allreduce_default_container_type) {
    Communicator<OwnContainer> comm;
    std::vector<int>           input = {comm.rank_signed(), 42};

    // This just has to compile
    OwnContainer<int> result = comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}));
}

TEST(AllreduceTest, send_recv_type_is_out_parameter) {
    Communicator           comm;
    std::vector<int> const data{1};
    MPI_Datatype           send_recv_type;
    auto recv_buf = comm.allreduce(send_buf(data), send_recv_type_out(send_recv_type), op(kamping::ops::plus<>{}));

    EXPECT_EQ(send_recv_type, MPI_INT);
    EXPECT_EQ(recv_buf.size(), 1);
    EXPECT_EQ(recv_buf.front(), comm.size());
}

TEST(AllreduceTest, send_recv_type_part_of_result_object) {
    Communicator           comm;
    std::vector<int> const data{1};
    auto                   result = comm.allreduce(send_buf(data), send_recv_type_out(), op(kamping::ops::plus<>{}));

    EXPECT_EQ(result.extract_send_recv_type(), MPI_INT);
    auto recv_buf = result.extract_recv_buffer();
    EXPECT_EQ(recv_buf.size(), 1);
    EXPECT_EQ(recv_buf.front(), comm.size());
}

// Death test + MPI does not work
/// @todo Add a proper test for the input validation.
// TEST(AllreduceTest, different_send_buf_sizes_fails) {
//     Communicator comm;
//
//     std::vector<int> input(comm.rank());
//     assert(input.size() == comm.rank());
//
//     if (kassert::internal::assertion_enabled(assert::light_communication)) {
//         EXPECT_KASSERT_FAILS(
//             comm.allreduce(send_buf(input), op(kamping::ops::plus<>{})),
//             "The send buffer has to be the same size on all ranks.");
//     }
// }

TEST(AllreduceTest, allreduce_single) {
    Communicator comm;

    int       input  = comm.rank_signed();
    int const result = comm.allreduce_single(send_buf(input), op(kamping::ops::plus<>{}));

    int expected_result = (comm.size_signed() * (comm.size_signed() - 1)) / 2;
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_single_with_temporary) {
    Communicator comm;

    int const result = comm.allreduce_single(send_buf(comm.rank_signed()), op(kamping::ops::plus<>{}));

    int expected_result = (comm.size_signed() * (comm.size_signed() - 1)) / 2;
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, structured_bindings_explicit_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t>       values{comm.rank(), comm.rank()};
    std::vector<std::uint64_t> const expected_recv_buffer(2, comm.size() * (comm.size() - 1) / 2);
    std::vector<std::uint64_t>       recv_buffer(2);
    auto const [send_recv_type, send_recv_count] = comm.allreduce(
        send_recv_type_out(),
        send_recv_count_out(),
        send_buf(values),
        recv_buf(recv_buffer),
        op(kamping::ops::plus<>{})
    );

    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
    EXPECT_EQ(recv_buffer, expected_recv_buffer);
}

TEST(AllreduceTest, structured_bindings_explicit_owning_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t>       values{comm.rank(), comm.rank()};
    std::vector<std::uint64_t> const expected_recv_buffer(2, comm.size() * (comm.size() - 1) / 2);
    std::vector<std::uint64_t>       tmp(2);
    auto const [send_recv_type, send_recv_count, recv_buffer] = comm.allreduce(
        send_recv_type_out(),
        send_recv_count_out(),
        send_buf(values),
        recv_buf(std::move(tmp)),
        op(kamping::ops::plus<>{})
    );

    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
    EXPECT_EQ(recv_buffer, expected_recv_buffer);
}

TEST(AllreduceTest, structured_bindings_implicit_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t>       values{comm.rank(), comm.rank()};
    std::vector<std::uint64_t> const expected_recv_buffer(2, comm.size() * (comm.size() - 1) / 2);
    {
        std::vector<std::uint64_t> tmp(2);
        auto const [recv_buffer, send_recv_type, send_recv_count] =
            comm.allreduce(send_recv_type_out(), send_recv_count_out(), send_buf(values), op(kamping::ops::plus<>{}));

        EXPECT_EQ(send_recv_count, 2);
        EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
    {
        // non-owning send_recv_type out buffer
        std::vector<std::uint64_t> tmp(2);
        MPI_Datatype               send_recv_type;
        auto const [recv_buffer, send_recv_count] = comm.allreduce(
            send_recv_type_out(send_recv_type),
            send_recv_count_out(),
            send_buf(values),
            op(kamping::ops::plus<>{})
        );

        EXPECT_EQ(send_recv_count, 2);
        EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
}

TEST(AllreduceTest, inplace_basic) {
    Communicator comm;

    std::vector<int> values = {comm.rank_signed(), comm.rank_signed()};
    comm.allreduce(send_recv_buf(values), op(kamping::ops::plus<>{}));

    std::vector<int> expected_recv_buffer = {
        comm.size_signed() * (comm.size_signed() - 1) / 2,
        comm.size_signed() * (comm.size_signed() - 1) / 2};
    EXPECT_EQ(values, expected_recv_buffer);
}

TEST(AllreduceTest, inplace_out_parameters) {
    Communicator comm;

    std::vector<int> values = {comm.rank_signed(), comm.rank_signed()};
    auto [count, type] =
        comm.allreduce(send_recv_buf(values), op(kamping::ops::plus<>{}), send_recv_count_out(), send_recv_type_out());

    EXPECT_EQ(count, 2);
    EXPECT_EQ(type, MPI_INT);

    std::vector<int> expected_recv_buffer = {
        comm.size_signed() * (comm.size_signed() - 1) / 2,
        comm.size_signed() * (comm.size_signed() - 1) / 2};
    EXPECT_EQ(values, expected_recv_buffer);
}

TEST(AllreduceTest, inplace_rvalue_buffer) {
    Communicator comm;

    std::vector<int> values = {comm.rank_signed(), comm.rank_signed()};
    auto             result = comm.allreduce(send_recv_buf(std::move(values)), op(kamping::ops::plus<>()));

    std::vector<int> expected_recv_buffer = {
        comm.size_signed() * (comm.size_signed() - 1) / 2,
        comm.size_signed() * (comm.size_signed() - 1) / 2};
    EXPECT_EQ(result, expected_recv_buffer);
}

TEST(AllreduceTest, inplace_explicit_count) {
    Communicator comm;

    std::vector<int> values = {comm.rank_signed(), -1};
    comm.allreduce(send_recv_buf(values), op(kamping::ops::plus<>{}), send_recv_count(1));

    std::vector<int> expected_recv_buffer = {comm.size_signed() * (comm.size_signed() - 1) / 2, -1};
    EXPECT_EQ(values, expected_recv_buffer);
}

TEST(AllreduceTest, inplace_explicit_type) {
    Communicator comm;

    std::vector<std::pair<int, int>> values = {std::pair{comm.rank_signed(), comm.rank_signed()}};
    MPI_Datatype                     type   = struct_type<std::pair<int, int>>::data_type();
    MPI_Type_commit(&type);
    comm.allreduce(
        send_recv_buf(values),
        op(
            [](std::pair<int, int> const& lhs, std::pair<int, int> const& rhs) {
                return std::pair{lhs.first + rhs.first, lhs.second + rhs.second};
            },
            kamping::ops::commutative
        ),
        send_recv_type(type),
        send_recv_count(1)
    );
    MPI_Type_free(&type);

    std::vector<std::pair<int, int>> expected_recv_buffer = {
        {comm.size_signed() * (comm.size_signed() - 1) / 2, comm.size_signed() * (comm.size_signed() - 1) / 2}};
    EXPECT_EQ(values, expected_recv_buffer);
}
