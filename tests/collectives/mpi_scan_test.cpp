
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

#include "gmock/gmock.h"

#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/scan.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(ScanTest, scan_single) {
    Communicator comm;

    int input = 42;

    auto result = comm.scan_single(send_buf(input), op(kamping::ops::plus<>{}));
    static_assert(std::is_same_v<decltype(result), decltype(input)>);
    int expected_result = (comm.rank_signed() + 1) * 42;
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_single_with_temporary) {
    Communicator comm;

    auto result          = comm.scan_single(send_buf(42), op(kamping::ops::plus<>{}));
    int  expected_result = (comm.rank_signed() + 1) * 42;
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_single_vector_of_size_1) {
    Communicator comm;

    std::vector<int> input = {42};

    auto result = comm.scan_single(send_buf(input.front()), op(kamping::ops::plus<>{}));
    static_assert(std::is_same_v<decltype(result), decltype(input)::value_type>);
    int expected_result = (comm.rank_signed() + 1) * 42;
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_explicit_send_recv_count_smaller_than_send_buffer_size) {
    Communicator comm;

    std::vector<int> input = {42, 1, 1, 1, 1};

    auto recv_buf = comm.scan(send_buf(input), send_recv_count(2), op(kamping::ops::plus<>{}));
    EXPECT_THAT(recv_buf, ElementsAre((comm.rank_signed() + 1) * 42, (comm.rank_signed() + 1)));
}

TEST(ScanTest, scan_explicit_send_recv_count_out_value_not_taken_into_account) {
    Communicator comm;

    std::vector<int> input           = {42, 1};
    int              send_recv_count = -1;

    auto recv_buf = comm.scan(send_buf(input), send_recv_count_out(send_recv_count), op(kamping::ops::plus<>{}));
    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(recv_buf, ElementsAre((comm.rank_signed() + 1) * 42, (comm.rank_signed() + 1)));
}

TEST(ScanTest, scan_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input = {42, 1};

    auto recv_buf = comm.scan(send_buf(input), send_recv_count(2), op(kamping::ops::plus<>{}));
    EXPECT_THAT(recv_buf, ElementsAre((comm.rank_signed() + 1) * 42, (comm.rank_signed() + 1)));
}

TEST(ScanTest, scan_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto result = comm.scan(send_buf(input), op(kamping::ops::plus<>{}));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {
        ((comm.rank_signed() + 1) * comm.rank_signed()) / 2,
        (comm.rank_signed() + 1) * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.scan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {
        ((comm.rank_signed() + 1) * comm.rank_signed()) / 2,
        (comm.rank_signed() + 1) * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_builtin_op_on_non_builtin_type) {
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

    auto result = comm.scan(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative));
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {
        ((comm.rank_signed() + 1) * comm.rank_signed()) / 2,
        (comm.rank_signed() + 1) * 42};
    EXPECT_EQ(result, expected_result);
}

int add_plus_42_function(int const& lhs, int const& rhs) {
    return lhs + rhs + 42;
}

TEST(ScanTest, scan_custom_operation_on_builtin_type) {
    Communicator comm;

    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };

    std::vector<int> input = {0, 17, 8};

    { // use function ptr
        auto result = comm.scan(send_buf(input), op(add_plus_42_function, kamping::ops::commutative));

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            (comm.rank_signed() + 1) * 0 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 17 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 8 + (comm.rank_signed()) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use lambda
        auto result = comm.scan(send_buf(input), op(add_plus_42_lambda, kamping::ops::commutative));

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            (comm.rank_signed() + 1) * 0 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 17 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 8 + (comm.rank_signed()) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use lambda inline
        auto result = comm.scan(
            send_buf(input),
            op([](auto const& lhs, auto const& rhs) { return lhs + rhs + 42; }, kamping::ops::commutative)
        );

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            (comm.rank_signed() + 1) * 0 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 17 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 8 + (comm.rank_signed()) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use function object
        struct MySum42 {
            int operator()(int const& lhs, int const& rhs) {
                return lhs + rhs + 42;
            }
        };
        auto result = comm.scan(send_buf(input), op(MySum42{}, kamping::ops::commutative));

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            (comm.rank_signed() + 1) * 0 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 17 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 8 + (comm.rank_signed()) * 42};
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ScanTest, scan_custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.scan(send_buf(input), op(get_right, kamping::ops::non_commutative));

    EXPECT_EQ(result.size(), 1);
    std::vector<int> expected_result = {comm.rank_signed() + 17};
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_custom_operation_on_custom_type) {
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

    Aggregate              agg1_expected   = {0, comm.rank_signed(), true, comm.rank_signed() + 1};
    Aggregate              agg2_expected   = {42, comm.rank_signed() + 42, false, comm.rank_signed() + 1};
    std::vector<Aggregate> expected_result = {agg1_expected, agg2_expected};

    auto result = comm.scan(send_buf(input), op(my_op, kamping::ops::commutative));

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_default_container_type) {
    Communicator<OwnContainer> comm;
    std::vector<int>           input = {comm.rank_signed(), 42};

    // This just has to compile
    OwnContainer<int> result = comm.scan(send_buf(input), op(kamping::ops::plus<>{}));
}

TEST(ScanTest, single_element_with_given_recv_buf_bigger_than_required) {
    Communicator     comm;
    std::vector<int> input               = {1};
    int              expected_recv_value = comm.rank_signed() + 1;

    {
        // recv buffer will be resized as policy is resize_to_fit
        std::vector<int> recv_buffer(2, -1);
        comm.scan(send_buf(input), recv_buf<resize_to_fit>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_EQ(recv_buffer.front(), expected_recv_value);
    }
    {
        // recv buffer will not be resized as it is large enough and policy is grow_only
        std::vector<int> recv_buffer(2, -1);
        comm.scan(send_buf(input), recv_buf<grow_only>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_THAT(recv_buffer, ElementsAre(expected_recv_value, -1));
    }
    {
        // recv buffer will not be resized as the policy is no_resize
        std::vector<int> recv_buffer(2, -1);
        comm.scan(send_buf(input), recv_buf<no_resize>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_THAT(recv_buffer, ElementsAre(expected_recv_value, -1));
    }
    {
        // recv buffer will not be resized as the policy is no_resize (default)
        std::vector<int> recv_buffer(2, -1);
        comm.scan(send_buf(input), recv_buf(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_THAT(recv_buffer, ElementsAre(expected_recv_value, -1));
    }
}

TEST(ScanTest, single_element_with_given_recv_buf_smaller_than_required) {
    Communicator     comm;
    std::vector<int> input = {1};
    std::vector<int> expected_recv_buffer{comm.rank_signed() + 1};

    {
        // recv buffer will be resized as policy is resize_to_fit
        std::vector<int> recv_buffer;
        comm.scan(send_buf(input), recv_buf<resize_to_fit>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
    {
        // recv buffer will be resized as policy is grow_only and buffer is too small
        std::vector<int> recv_buffer;
        comm.scan(send_buf(input), recv_buf<grow_only>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    {
        // recv buffer will not be resized as the policy is no_resize
        std::vector<int> recv_buffer;
        EXPECT_KASSERT_FAILS(
            comm.scan(send_buf(input), recv_buf<no_resize>(recv_buffer), op(kamping::ops::plus<>{})),
            ""
        );
    }
    {
        // recv buffer will not be resized as the policy is no_resize (default)
        std::vector<int> recv_buffer;
        EXPECT_KASSERT_FAILS(
            comm.scan(send_buf(input), recv_buf<no_resize>(recv_buffer), op(kamping::ops::plus<>{})),
            ""
        );
    }
#endif
}

TEST(ScanTest, send_recv_count_is_out_parameter) {
    Communicator     comm;
    std::vector<int> data{0, 1};
    int              send_recv_count = -1;
    auto result = comm.scan(send_buf(data), send_recv_count_out(send_recv_count), op(kamping::ops::plus<>{}));

    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(result, ElementsAre(0, comm.rank() + 1));
}

TEST(ScanTest, send_recv_count_is_part_of_result_object) {
    Communicator     comm;
    std::vector<int> data{0, 1};
    auto             result = comm.scan(send_buf(data), send_recv_count_out(), op(kamping::ops::plus<>{}));

    EXPECT_EQ(result.extract_send_recv_count(), 2);
    EXPECT_THAT(result.extract_recv_buffer(), ElementsAre(0, comm.rank() + 1));
}

TEST(ScanTest, send_recv_type_is_out_parameter) {
    Communicator     comm;
    std::vector<int> data{0, 1};
    MPI_Datatype     send_recv_type;
    auto             result =
        comm.scan(send_buf(data), send_recv_count(2), op(kamping::ops::plus<>{}), send_recv_type_out(send_recv_type));

    EXPECT_EQ(send_recv_type, MPI_INT);
    EXPECT_THAT(result, ElementsAre(0, comm.rank() + 1));
}

TEST(ScanTest, send_recv_type_is_part_of_result_object) {
    Communicator     comm;
    std::vector<int> data{0, 1};
    auto result = comm.scan(send_buf(data), send_recv_count(2), op(kamping::ops::plus<>{}), send_recv_type_out());

    EXPECT_EQ(result.extract_send_recv_type(), MPI_INT);
    EXPECT_THAT(result.extract_recv_buffer(), ElementsAre(0, comm.rank() + 1));
}

TEST(ScanTest, custom_operation_on_custom_mpi_type) {
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

    Aggregate              agg1_expected   = {0, dont_care, comm.rank_signed()};
    Aggregate              agg2_expected   = {42, dont_care, comm.rank_signed() + 42};
    std::vector<Aggregate> expected_result = {agg1_expected, agg2_expected};
    std::vector<Aggregate> recv_buffer(2);

    MPI_Type_commit(&int_padding_int);
    comm.scan(
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

TEST(ScanTest, custom_operation_on_custom_mpi_without_matching_cpp_type) {
    Communicator comm;
    int const    dont_care = -1;

    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input = {comm.rank_signed(), dont_care, dont_care, comm.rank_signed() + 42, dont_care, dont_care};

    int const        sum_of_smaller_ranks_inclusive = comm.rank_signed() * (comm.rank_signed() + 1) / 2;
    std::vector<int> expected_result                = {
                       sum_of_smaller_ranks_inclusive,
                       dont_care,
                       dont_care,
                       sum_of_smaller_ranks_inclusive + (comm.rank_signed() + 1) * 42,
                       dont_care,
                       dont_care};
    std::vector<int> recv_buffer(6, dont_care);

    MPI_Op user_defined_op;
    MPI_Op_create(sum_for_int_padding_padding_type, 1, &user_defined_op);
    MPI_Type_commit(&int_padding_padding);
    comm.scan(
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

TEST(ScanTest, structured_bindings_explicit_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t> input = {42u, 1u};
    std::vector<std::uint64_t> recv_buffer(2);

    auto [send_recv_count] =
        comm.scan(send_buf(input), send_recv_count_out(), op(kamping::ops::plus<>{}), recv_buf(recv_buffer));
    EXPECT_THAT(recv_buffer, ElementsAre((comm.rank() + 1) * 42, (comm.rank() + 1)));
    EXPECT_EQ(send_recv_count, 2);
}

TEST(ScanTest, structured_bindings_explicit_owning_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t> input = {42u, 1u};

    auto [send_recv_count, recv_buffer] = comm.scan(
        send_buf(input),
        send_recv_count_out(),
        op(kamping::ops::plus<>{}),
        recv_buf(alloc_new<std::vector<std::uint64_t>>)
    );
    EXPECT_THAT(recv_buffer, ElementsAre((comm.rank() + 1) * 42, (comm.rank() + 1)));
    EXPECT_EQ(send_recv_count, 2);
}

TEST(ScanTest, structured_bindings_implicit_recv_buffer) {
    Communicator comm;

    std::vector<std::uint64_t> input = {42u, 1u};

    auto [recv_buffer, send_recv_type, send_recv_count] =
        comm.scan(send_recv_type_out(), send_buf(input), send_recv_count_out(), op(kamping::ops::plus<>{}));
    EXPECT_THAT(recv_buffer, ElementsAre((comm.rank() + 1) * 42, (comm.rank() + 1)));
    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(possible_mpi_datatypes<std::uint64_t>(), Contains(send_recv_type));
}

TEST(ScanTest, inplace_basic) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    comm.scan(send_recv_buf(data), op(kamping::ops::plus<>{}));
    EXPECT_THAT(data, ElementsAre((comm.rank() + 1) * 42, (comm.rank() + 1)));
}

TEST(ScanTest, inplace_out_parameters) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    auto [count, type] =
        comm.scan(send_recv_buf(data), send_recv_count_out(), send_recv_type_out(), op(kamping::ops::plus<>{}));
    EXPECT_EQ(count, 2);
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(type));
    EXPECT_THAT(data, ElementsAre((comm.rank() + 1) * 42, (comm.rank() + 1)));
}

TEST(ScanTest, inplace_rvalue_buffer) {
    Communicator comm;

    auto result = comm.scan(send_recv_buf(std::vector<int>{42, 1}), op(kamping::ops::plus<>{}));
    EXPECT_THAT(result, ElementsAre((comm.rank() + 1) * 42, (comm.rank() + 1)));
}

TEST(ScanTest, inplace_explicit_count) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    comm.scan(send_recv_buf(data), send_recv_count(1), op(kamping::ops::plus<>{}));
    EXPECT_THAT(data, ElementsAre((comm.rank() + 1) * 42, 1 /* unchanged */));
}

TEST(ScanTest, inplace_explicit_count_resize) {
    Communicator comm;

    std::vector<int> data = {42, 1};
    comm.scan(send_recv_buf<resize_to_fit>(data), send_recv_count(1), op(kamping::ops::plus<>{}));
    EXPECT_THAT(data, ElementsAre((comm.rank() + 1) * 42));
}

TEST(ScanTest, inplace_explicit_type) {
    Communicator comm;

    std::pair<int, int> data = {42, 1};
    MPI_Datatype        type = struct_type<std::pair<int, int>>::data_type();
    MPI_Type_commit(&type);
    comm.scan(
        send_recv_buf(data),
        send_recv_count(1),
        op([](auto const& lhs,
              auto const& rhs) { return std::make_pair(lhs.first + rhs.first, lhs.second + rhs.second); },
           kamping::ops::commutative),
        send_recv_type(type)
    );
    MPI_Type_free(&type);
    EXPECT_EQ(data, std::make_pair((comm.rank_signed() + 1) * 42, comm.rank_signed() + 1));
}
