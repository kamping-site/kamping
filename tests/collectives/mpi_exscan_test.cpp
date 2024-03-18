// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
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
#include "kamping/collectives/exscan.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(ExscanTest, exscan_single) {
    Communicator comm;

    int input = 42;

    auto result = comm.exscan_single(send_buf(input), op(kamping::ops::plus<>{}));
    static_assert(std::is_same_v<decltype(result), decltype(input)>);
    if (comm.rank() != 0) {
        int expected_result = comm.rank_signed() * 42;
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, scan_single_with_temporary) {
    Communicator comm;

    auto result          = comm.exscan_single(send_buf(42), op(kamping::ops::plus<>{}));
    int  expected_result = comm.rank_signed() * 42;
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, exscan_single_values_on_rank_0) {
    Communicator comm;

    int input = 42;

    int result          = comm.exscan_single(send_buf(input), op(kamping::ops::plus<>{}), values_on_rank_0(0));
    int expected_result = comm.rank_signed() * 42;
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, exscan_single_vector_of_size_1) {
    Communicator comm;

    std::vector<int> input = {42};

    auto result = comm.exscan_single(send_buf(input.front()), op(kamping::ops::plus<>{}));
    static_assert(std::is_same_v<decltype(result), decltype(input)::value_type>);
    if (comm.rank() != 0) {
        int expected_result = comm.rank_signed() * 42;
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto result = comm.exscan(send_buf(input), op(kamping::ops::plus<>{}));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, with_receive_buffer_and_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input = {1, 2};
    std::vector<int> result;

    comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), send_recv_count(1), recv_buf<resize_to_fit>(result));
    EXPECT_THAT(result, ElementsAre(comm.rank_signed()));
}

TEST(ExscanTest, with_receive_buffer_send_recv_count_out) {
    // tests that send_recv_count_out is not used to determine the number of elements to send.
    Communicator comm;

    std::vector<int> input = {1};
    std::vector<int> result;
    int              send_recv_count_value = -1;

    comm.exscan(
        send_buf(input),
        op(kamping::ops::plus<>{}),
        send_recv_count_out(send_recv_count_value),
        recv_buf<resize_to_fit>(result)
    );
    EXPECT_EQ(send_recv_count_value, 1);
    EXPECT_THAT(result, ElementsAre(comm.rank_signed()));
}

TEST(ExscanTest, recv_buffer_is_given_and_larger_than_required) {
    Communicator comm;

    std::vector<int> input = {1};

    {
        std::vector<int> result(2, -1);
        // recv buffer will be resizes to size 1 as policy is resize_to_fit
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result));
        EXPECT_THAT(result, ElementsAre(comm.rank_signed()));
    }
    {
        std::vector<int> result(2, -1);
        // recv buffer will not be resizes as it large enough and resize policy is grow_only
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<grow_only>(result));
        EXPECT_THAT(result, ElementsAre(comm.rank_signed(), -1));
    }
    {
        std::vector<int> result(2, -1);
        // recv buffer will not be resizes as resize policy is no resize
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<no_resize>(result));
        EXPECT_THAT(result, ElementsAre(comm.rank_signed(), -1));
    }
}

TEST(ExscanTest, recv_buffer_is_given_and_smaller_than_required) {
    Communicator comm;

    std::vector<int> const input = {1};

    {
        std::vector<int> result;
        // recv buffer will be resizes to size 1 as policy is resize_to_fit
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result));
        EXPECT_THAT(result, ElementsAre(comm.rank_signed()));
    }
    {
        std::vector<int> result;
        // recv buffer will be resizes to size 1 as policy is grow_only and given underlying buffer is too small
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<grow_only>(result));
        EXPECT_THAT(result, ElementsAre(comm.rank_signed()));
    }
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    {
        std::vector<int> result;
        // recv buffer will not be resizes as resize policy is no resize
        EXPECT_KASSERT_FAILS(comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<no_resize>(result)), "");
    }
#endif
}

TEST(ExscanTest, builtin_op_on_non_builtin_type) {
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

    auto result =
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative), values_on_rank_0(MyInt{0}));
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, identity_not_auto_deducible_and_no_values_on_rank_0_provided) {
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

    auto result = comm.exscan(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative));
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    if (comm.rank() != 0) { // The result of this exscan() is not defined on rank 0.
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, non_identity_values_on_rank_0) {
    Communicator comm;

    std::vector<int> input = {0};

    auto result = comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), values_on_rank_0(1337));
    EXPECT_EQ(result.size(), 1);

    std::vector<int> expected_result;
    if (comm.rank() == 0) {
        expected_result.push_back(1337);
    } else {
        expected_result.push_back(0);
    }
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, non_identity_value_on_rank_0_with_given_recv_buffer_bigger_than_required) {
    // ensure that the postprocessing on rank 0 does not rely on the size of the recv buffer but the (auto-deduced)
    // send_recv_count
    Communicator comm;

    std::vector<int> input = {0};
    std::vector<int> result{-1, -1}; // bigger than required
    int const        default_value_on_rank_0 = 1337;

    comm.exscan(
        send_buf(input),
        op(kamping::ops::plus<>{}),
        values_on_rank_0(default_value_on_rank_0),
        recv_buf<no_resize>(result)
    );
    if (comm.rank() == 0) {
        EXPECT_THAT(result, ElementsAre(default_value_on_rank_0, -1));
    } else {
        EXPECT_THAT(result, ElementsAre(0, -1));
    }
}

TEST(ExscanTest, non_identity_values_on_rank_0_with_given_recv_buffer_bigger_than_required) {
    // ensure that the postprocessing on rank 0 does not rely on the size of the recv buffer but the (auto-deduced)
    // send_recv_count
    Communicator comm;

    std::vector<int> const input = {0, 0};
    std::vector<int>       result{-1, -1, -1, -1}; // bigger than required
    int const              default_value            = 1337;
    std::vector<int> const default_values_on_rank_0 = {default_value, default_value};

    comm.exscan(
        send_buf(input),
        op(kamping::ops::plus<>{}),
        values_on_rank_0(default_values_on_rank_0),
        recv_buf<no_resize>(result)
    );

    if (comm.rank() == 0) {
        EXPECT_THAT(result, ElementsAre(default_value, default_value, -1, -1));
    } else {
        EXPECT_THAT(result, ElementsAre(0, 0, -1, -1));
    }
}

int add_plus_42_function(int const& lhs, int const& rhs) {
    return lhs + rhs + 42;
}

TEST(ExscanTest, custom_operation_on_builtin_type) {
    Communicator comm;

    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };

    std::vector<int> input = {0, 17, 8};

    { // use function ptr
        auto result = comm.exscan(
            send_buf(input),
            op(add_plus_42_function, kamping::ops::commutative),
            values_on_rank_0({0, 1, 2})
        );

        EXPECT_EQ(result.size(), 3);
        if (comm.rank() == 0) {
            EXPECT_EQ(result, (std::vector<int>{0, 1, 2}));
        } else {
            std::vector<int> expected_result = {
                comm.rank_signed() * 0 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 17 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 8 + (comm.rank_signed() - 1) * 42};
            EXPECT_EQ(result, expected_result);
        }
    }

    { // use lambda
        auto result = comm.exscan(
            send_buf(input),
            op(add_plus_42_lambda, kamping::ops::commutative),
            values_on_rank_0({0, 1, 2})
        );

        EXPECT_EQ(result.size(), 3);
        if (comm.rank() == 0) {
            EXPECT_EQ(result, (std::vector<int>{0, 1, 2}));
        } else {
            std::vector<int> expected_result = {
                comm.rank_signed() * 0 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 17 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 8 + (comm.rank_signed() - 1) * 42};
            EXPECT_EQ(result, expected_result);
        }
    }

    { // use lambda inline
        auto result = comm.exscan(
            send_buf(input),
            op([](auto const& lhs, auto const& rhs) { return lhs + rhs + 42; }, kamping::ops::commutative),
            values_on_rank_0({0, 1, 2})
        );

        EXPECT_EQ(result.size(), 3);
        if (comm.rank() == 0) {
            EXPECT_EQ(result, (std::vector<int>{0, 1, 2}));
        } else {
            std::vector<int> expected_result = {
                comm.rank_signed() * 0 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 17 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 8 + (comm.rank_signed() - 1) * 42};
            EXPECT_EQ(result, expected_result);
        }
    }

    { // use function object
        struct MySum42 {
            int operator()(int const& lhs, int const& rhs) {
                return lhs + rhs + 42;
            }
        };
        auto result =
            comm.exscan(send_buf(input), op(MySum42{}, kamping::ops::commutative), values_on_rank_0({0, 1, 2}));

        EXPECT_EQ(result.size(), 3);
        if (comm.rank() == 0) {
            EXPECT_EQ(result, (std::vector<int>{0, 1, 2}));
        } else {
            std::vector<int> expected_result = {
                comm.rank_signed() * 0 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 17 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 8 + (comm.rank_signed() - 1) * 42};
            EXPECT_EQ(result, expected_result);
        }
    }
}

TEST(ExscanTest, custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.exscan(send_buf(input), op(get_right, kamping::ops::non_commutative), values_on_rank_0(0));

    EXPECT_EQ(result.size(), 1);
    if (comm.rank() == 0) {
        EXPECT_EQ(result, (std::vector<int>{0}));
    } else {
        std::vector<int> expected_result = {comm.rank_signed() - 1 + 17};
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, default_container_type) {
    Communicator<OwnContainer> comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    // This just has to compile
    OwnContainer<int> result = comm.exscan(send_buf(input), op(kamping::ops::plus<>{}));
}

TEST(ExscanTest, given_values_on_rank_0_have_wrong_size) {
    Communicator comm;

    std::vector<int> input = {0, 0};
    std::vector<int> result{-1, -1};

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    // test kassert that ensure that size of values_on_rank_0 buffer is either 1 or matches the send_recv_count
    if (comm.rank() == 0) {
        EXPECT_KASSERT_FAILS(
            comm.exscan(
                send_buf(input),
                op(kamping::ops::plus<>{}),
                values_on_rank_0({-1, -1, -1, -1}),
                recv_buf<no_resize>(result)
            ),
            ""
        );
    } else {
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<no_resize>(result));
    }
#endif
}

TEST(ExscanTest, send_recv_count_is_out_parameter) {
    Communicator     comm;
    std::vector<int> data{0, 1};
    int              send_recv_count = -1;
    auto             result          = comm.exscan(
        send_buf(data),
        send_recv_count_out(send_recv_count),
        op(kamping::ops::plus<>{}),
        values_on_rank_0({0, 0})
    );

    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(result, ElementsAre(0, comm.rank()));
}

TEST(ExscanTest, send_recv_count_is_part_of_result_object) {
    Communicator     comm;
    std::vector<int> data{0, 1};
    auto             result =
        comm.exscan(send_buf(data), send_recv_count_out(), op(kamping::ops::plus<>{}), values_on_rank_0({0, 0}));

    EXPECT_EQ(result.extract_send_recv_count(), 2);
    EXPECT_THAT(result.extract_recv_buffer(), ElementsAre(0, comm.rank()));
}

TEST(ExscanTest, send_recv_type_is_out_parameter) {
    Communicator     comm;
    std::vector<int> data{0, 1};
    MPI_Datatype     send_recv_type;
    auto             result = comm.exscan(
        send_buf(data),
        send_recv_count(2),
        op(kamping::ops::plus<>{}),
        send_recv_type_out(send_recv_type),
        values_on_rank_0({0, 0})
    );

    EXPECT_EQ(send_recv_type, MPI_INT);
    EXPECT_THAT(result, ElementsAre(0, comm.rank()));
}

TEST(ExscanTest, send_recv_type_is_part_of_result_object) {
    Communicator     comm;
    std::vector<int> data{0, 1};
    auto             result = comm.exscan(
        send_buf(data),
        send_recv_count(2),
        op(kamping::ops::plus<>{}),
        send_recv_type_out(),
        values_on_rank_0({0, 0})
    );

    EXPECT_EQ(result.extract_send_recv_type(), MPI_INT);
    EXPECT_THAT(result.extract_recv_buffer(), ElementsAre(0, comm.rank()));
}

TEST(ExscanTest, custom_operation_on_custom_mpi_type) {
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

    Aggregate              agg1_expected   = {0, dont_care, std::max(comm.rank_signed() - 1, 0)};
    Aggregate              agg2_expected   = {42, dont_care, std::max(comm.rank_signed() - 1 + 42, 42)};
    std::vector<Aggregate> expected_result = {agg1_expected, agg2_expected};
    std::vector<Aggregate> recv_buffer(2);

    MPI_Type_commit(&int_padding_int);
    comm.exscan(
        send_buf(input),
        send_recv_count(2),
        send_recv_type(int_padding_int),
        op(my_op, kamping::ops::commutative),
        recv_buf<no_resize>(recv_buffer),
        values_on_rank_0({Aggregate{0, dont_care, 0}, Aggregate{42, dont_care, 42}})
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

TEST(ExscanTest, custom_operation_on_custom_mpi_without_matching_cpp_type) {
    Communicator comm;
    int const    dont_care = -1;

    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input = {comm.rank_signed(), dont_care, dont_care, comm.rank_signed() + 42, dont_care, dont_care};

    int const        sum_of_smaller_ranks_exclusive = comm.rank_signed() * (comm.rank_signed() - 1) / 2;
    std::vector<int> expected_result                = {
                       sum_of_smaller_ranks_exclusive,
                       dont_care,
                       dont_care,
                       sum_of_smaller_ranks_exclusive + (comm.rank_signed()) * 42,
                       dont_care,
                       dont_care};
    std::vector<int> recv_buffer(6, dont_care);

    MPI_Op user_defined_op;
    MPI_Op_create(sum_for_int_padding_padding_type, 1, &user_defined_op);
    MPI_Type_commit(&int_padding_padding);
    comm.exscan(
        send_buf(input),
        send_recv_count(2),
        send_recv_type(int_padding_padding),
        op(user_defined_op),
        recv_buf<no_resize>(recv_buffer),
        values_on_rank_0({0, dont_care, dont_care, 0, dont_care, dont_care})
    );
    MPI_Type_free(&int_padding_padding);
    MPI_Op_free(&user_defined_op);

    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(
    ExscanTest, custom_operation_on_user_provided_datatype_mpi_without_different_value_types_for_send_and_recv_buffer
) {
    Communicator comm;
    int const    dont_care = -1;

    struct ThreeInts {
        int value;
        int padding[2];
    };

    MPI_Datatype           int_padding_padding = MPI_INT_padding_padding();
    std::vector<ThreeInts> input               = {
                      ThreeInts{comm.rank_signed(), {dont_care, dont_care}},
                      ThreeInts{comm.rank_signed() + 42, {dont_care, dont_care}}};

    int const        sum_of_smaller_ranks_exclusive = comm.rank_signed() * (comm.rank_signed() - 1) / 2;
    std::vector<int> expected_result                = {
                       sum_of_smaller_ranks_exclusive,
                       dont_care,
                       dont_care,
                       sum_of_smaller_ranks_exclusive + (comm.rank_signed()) * 42,
                       dont_care,
                       dont_care};
    std::vector<int> recv_buffer(6, dont_care);

    MPI_Op user_defined_op;
    MPI_Op_create(sum_for_int_padding_padding_type, 1, &user_defined_op);
    MPI_Type_commit(&int_padding_padding);
    comm.exscan(
        send_buf(input),
        send_recv_count(2),
        send_recv_type(int_padding_padding),
        op(user_defined_op),
        recv_buf<no_resize>(recv_buffer),
        values_on_rank_0({0, dont_care, dont_care, 0, dont_care, dont_care})
    );
    MPI_Type_free(&int_padding_padding);
    MPI_Op_free(&user_defined_op);
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(ExscanTest, structured_bindings_explicit_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t> input = {42u, 1u};
    std::vector<std::uint64_t> recv_buffer(2);

    auto [send_recv_count] =
        comm.exscan(send_buf(input), send_recv_count_out(), op(kamping::ops::plus<>{}), recv_buf(recv_buffer));
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank() * 42, comm.rank()));
    EXPECT_EQ(send_recv_count, 2);
}

TEST(ExscanTest, structured_bindings_explicit_owning_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t> input = {42u, 1u};

    auto [send_recv_count, recv_buffer] = comm.exscan(
        send_buf(input),
        send_recv_count_out(),
        op(kamping::ops::plus<>{}),
        recv_buf(alloc_new<std::vector<std::uint64_t>>)
    );
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank() * 42, comm.rank()));
    EXPECT_EQ(send_recv_count, 2);
}

TEST(ExscanTest, structured_bindings_implicit_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t> input = {42u, 1u};

    auto [recv_buffer, send_recv_type, send_recv_count] =
        comm.exscan(send_recv_type_out(), send_buf(input), send_recv_count_out(), op(kamping::ops::plus<>{}));
    EXPECT_THAT(recv_buffer, ElementsAre(comm.rank() * 42, comm.rank()));
    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
}

TEST(ExscanTest, inplace_basic) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    comm.exscan(send_recv_buf(data), op(kamping::ops::plus<>{}));
    EXPECT_THAT(data, ElementsAre(comm.rank() * 42, comm.rank()));
}

TEST(ExscanTest, inplace_out_parameters) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    auto [count, type] =
        comm.exscan(send_recv_buf(data), op(kamping::ops::plus<>{}), send_recv_count_out(), send_recv_type_out());
    EXPECT_EQ(count, 2);
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(type));
    EXPECT_THAT(data, ElementsAre(comm.rank() * 42, comm.rank()));
}

TEST(ExscanTest, inplace_rvalue_buffer) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    data                  = comm.exscan(send_recv_buf(std::move(data)), op(kamping::ops::plus<>{}));
    EXPECT_THAT(data, ElementsAre(comm.rank() * 42, comm.rank()));
}

TEST(ExscanTest, inplace_explicit_count) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    comm.exscan(send_recv_buf(data), send_recv_count(1), op(kamping::ops::plus<>{}));
    EXPECT_THAT(data, ElementsAre(comm.rank() * 42, 1 /*unchanged*/));
}

TEST(ExscanTest, inplace_explicit_count_resize) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    comm.exscan(send_recv_buf<resize_to_fit>(data), send_recv_count(1), op(kamping::ops::plus<>{}));
    EXPECT_THAT(data, ElementsAre(comm.rank() * 42));
}

TEST(ExscanTest, inplace_explicit_type) {
    Communicator comm;

    std::pair<int, int> data = {42, 1};
    MPI_Datatype        type = struct_type<std::pair<int, int>>::data_type();
    MPI_Type_commit(&type);
    comm.exscan(
        send_recv_buf(data),
        send_recv_count(1),
        op([](auto const& lhs,
              auto const& rhs) { return std::make_pair(lhs.first + rhs.first, lhs.second + rhs.second); },
           kamping::ops::commutative),
        send_recv_type(type),
        values_on_rank_0(std::pair{0, 0})
    );
    MPI_Type_free(&type);
    EXPECT_EQ(data, std::make_pair(comm.rank_signed() * 42, comm.rank_signed()));
}
