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

#include <cstddef>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/allgather.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(AllgatherTest, allgather_single_element_no_receive_buffer) {
    Communicator comm;
    auto         value = comm.rank();

    auto result = comm.allgather(send_buf(value));
    EXPECT_EQ(comm.root(), 0);
    ASSERT_EQ(result.size(), comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        EXPECT_EQ(result[i], i);
    }

    // Change default root and test with communicator's default root again, this should not change anything.
    comm.root(comm.size_signed() - 1);
    result = comm.allgather(send_buf(value));
    EXPECT_EQ(comm.root(), comm.size() - 1);
    ASSERT_EQ(result.size(), comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        EXPECT_EQ(result[i], i);
    }
}

TEST(AllgatherTest, allgather_single_custom_element_no_receive_buffer) {
    Communicator comm;
    struct CustomDataType {
        int rank;
        int additional_value;
    }; // struct custom_data_type

    CustomDataType value = {comm.rank_signed(), comm.size_signed() - comm.rank_signed()};

    auto result = comm.allgather(send_buf(value));
    ASSERT_EQ(result.size(), comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        EXPECT_EQ(result[i].rank, i);
        EXPECT_EQ(result[i].additional_value, comm.size() - i);
    }
}

TEST(AllgatherTest, allgather_single_element_with_receive_buffer) {
    Communicator                 comm;
    auto                         value = comm.rank();
    std::vector<decltype(value)> result(0);

    comm.allgather(send_buf(value), recv_buf<resize_to_fit>(result));
    ASSERT_EQ(result.size(), comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        EXPECT_EQ(result[i], i);
    }
}

TEST(AllgatherTest, allgather_single_element_with_explicit_send_and_recv_count) {
    Communicator           comm;
    std::vector<int> const data(5, comm.rank_signed());
    int const              send_count_value = 1;
    int const              recv_count_value = 1;

    {
        // test that send_count parameter overwrites automatic deduction of send counts from the size of the send buffer
        auto recv_buf = comm.allgather(send_buf(data), send_count(send_count_value));
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(recv_buf[i], i);
        }
    }
    {
        // test that recv_count parameter overwrites automatic deduction of recv counts from the size of the send
        // counts. Currently these two values must be identical, as we do not yet support custom mpi datatypes where
        // send and recv counts may differ due to different send/recv types. // TODO adapt comment once custom mpi
        // datatypes are supported.
        auto recv_buf = comm.allgather(send_buf(data), send_count(send_count_value), recv_count(recv_count_value));
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(recv_buf[i], i);
        }
    }
}

TEST(AllgatherTest, allgather_single_element_with_r_values_in_send_and_recv_count_out) {
    Communicator           comm;
    std::vector<int> const data{comm.rank_signed()};
    // the values in send_counts_out, recv_counts_out should be ignored as they merely provide "storage" for the
    // values computed by kamping. (A mechanism which is not that useful for plain integers)
    std::vector<int> expected_recv_buf(comm.size());
    std::iota(expected_recv_buf.begin(), expected_recv_buf.end(), 0);

    {
        // extract methods
        auto result = comm.allgather(send_buf(data), send_count_out(), recv_count_out());
        EXPECT_EQ(result.extract_recv_buffer(), expected_recv_buf);
        EXPECT_EQ(result.extract_send_count(), 1);
        EXPECT_EQ(result.extract_recv_count(), 1);
    }
    {
        // structured binding
        auto [recv_buf, send_count, recv_count] = comm.allgather(send_buf(data), send_count_out(), recv_count_out());

        EXPECT_EQ(recv_buf, expected_recv_buf);
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(recv_count, 1);
    }
}

TEST(AllgatherTest, allgather_single_element_with_l_values_in_send_and_recv_count_out) {
    Communicator           comm;
    std::vector<int> const data{comm.rank_signed()};
    {
        // the values in send_counts_out, recv_counts_out should be ignored
        int  send_count = -1;
        int  recv_count = -1;
        auto recv_buf   = comm.allgather(send_buf(data), send_count_out(send_count), recv_count_out(recv_count));
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(recv_count, 1);
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(recv_buf[i], i);
        }
    }
}

TEST(AllgatherTest, allgather_single_element_with_given_recv_buf_bigger_than_required) {
    Communicator           comm;
    std::vector<int> const data{comm.rank_signed()};
    std::vector<int>       expected_recv_buffer(comm.size());
    std::iota(expected_recv_buffer.begin(), expected_recv_buffer.end(), 0);

    {
        // recv buffer will be resized to the size of the communicator
        std::vector<int> recv_buffer(2 * comm.size());
        comm.allgather(send_buf(data), recv_buf<resize_to_fit>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), comm.size());
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
    {
        // recv buffer will not be resized as it is large enough and policy is grow_only
        std::vector<int> recv_buffer(2 * comm.size());
        comm.allgather(send_buf(data), recv_buf<grow_only>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
    }
    {
        // recv buffer will not be resized as the policy is no_resize
        std::vector<int> recv_buffer(2 * comm.size());
        comm.allgather(send_buf(data), recv_buf<no_resize>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
    }
    {
        // recv buffer will not be resized as the policy is no_resize (default)
        std::vector<int> recv_buffer(2 * comm.size());
        comm.allgather(send_buf(data), recv_buf(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
    }
}

TEST(AllgatherTest, given_recv_buffer_smaller_than_required) {
    Communicator           comm;
    std::vector<int> const data{comm.rank_signed()};
    std::vector<int>       expected_recv_buffer(comm.size());
    std::iota(expected_recv_buffer.begin(), expected_recv_buffer.end(), 0);

    {
        // recv buffer will be resized to the size of the communicator
        std::vector<int> recv_buffer(comm.size() / 2);
        comm.allgather(send_buf(data), recv_buf<resize_to_fit>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), comm.size());
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
    {
        // recv buffer will not be resized as it is large enough and policy is grow_only
        std::vector<int> recv_buffer(comm.size() / 2);
        comm.allgather(send_buf(data), recv_buf<grow_only>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), comm.size());
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
}
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST(AllgatherTest, given_recv_buffer_smaller_than_required_with_policy_no_resize) {
    Communicator comm;

    std::vector<int> input{comm.rank_signed()};
    std::vector<int> recv_buffer;
    // test kassert for sufficient size of recv buffer
    EXPECT_KASSERT_FAILS(comm.allgather(send_buf(input), recv_buf<no_resize>(recv_buffer)), "");
    // same test but this time without explicit no_resize for the recv buffer as this is the default resize
    // policy
    EXPECT_KASSERT_FAILS(comm.allgather(send_buf(input), recv_buf(recv_buffer)), "");
}
#endif

TEST(AllgatherTest, allgather_multiple_elements_no_receive_buffer) {
    Communicator     comm;
    std::vector<int> values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    auto             result = comm.allgather(send_buf(values));

    EXPECT_EQ(result.size(), values.size() * comm.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], i / values.size());
    }
}

TEST(AllgatherTest, allgather_multiple_elements_with_receive_buffer) {
    Communicator     comm;
    std::vector<int> values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    std::vector<int> result(0);

    comm.allgather(send_buf(values), recv_buf<resize_to_fit>(result));

    EXPECT_EQ(result.size(), values.size() * comm.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], i / values.size());
    }
}

TEST(AllgatherTest, allgather_receive_custom_container) {
    Communicator      comm;
    std::vector<int>  values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    OwnContainer<int> result;

    comm.allgather(send_buf(values), recv_buf<resize_to_fit>(result));

    EXPECT_EQ(result.size(), values.size() * comm.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], i / values.size());
    }
}

TEST(AllgatherTest, allgather_send_custom_container) {
    Communicator      comm;
    OwnContainer<int> values(4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = comm.rank_signed();
    }
    std::vector<int> result;

    comm.allgather(send_buf(values), recv_buf<resize_to_fit>(result));

    EXPECT_EQ(result.size(), values.size() * comm.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], i / values.size());
    }
}

TEST(AllgatherTest, allgather_send_and_receive_custom_container) {
    Communicator      comm;
    OwnContainer<int> values(4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = comm.rank_signed();
    }
    OwnContainer<int> result;

    comm.allgather(send_buf(values), recv_buf<resize_to_fit>(result));

    EXPECT_EQ(result.size(), values.size() * comm.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], i / values.size());
    }
}

TEST(AllgatherTest, allgather_single_element_initializer_list_bool_no_receive_buffer) {
    Communicator comm;
    // gather does not support single element bool when specifying no recv_buffer, because the default receive buffer is
    // std::vector<bool>, which is not supported
    auto result = comm.allgather(send_buf({false}));
    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), comm.size());
    for (auto elem: result) {
        EXPECT_EQ(elem, false);
    }
}

TEST(AllgatherTest, allgather_initializer_list_bool_no_receive_buffer) {
    Communicator comm;
    auto         result = comm.allgather(send_buf({false, false}));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), 2 * comm.size());
    for (auto elem: result) {
        EXPECT_EQ(elem, false);
    }
}

TEST(AllgatherTest, allgather_single_element_kabool_no_receive_buffer) {
    Communicator comm;
    auto         result = comm.allgather(send_buf(kabool{false}));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), comm.size());
    for (auto elem: result) {
        EXPECT_EQ(elem, false);
    }
}

TEST(AllgatherTest, allgather_single_element_bool_with_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> result;
    comm.allgather(send_buf({false}), recv_buf<resize_to_fit>(result));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), comm.size());
    for (auto elem: result) {
        EXPECT_EQ(elem, false);
    }
}

TEST(AllgatherTest, allgather_single_element_kabool_with_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> result;
    comm.allgather(send_buf(kabool{false}), recv_buf<resize_to_fit>(result));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), comm.size());
    for (auto elem: result) {
        EXPECT_EQ(elem, false);
    }
}

TEST(AllgatherTest, allgather_multiple_elements_kabool_no_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> input  = {false, true};
    auto                result = comm.allgather(send_buf(input));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), 2 * comm.size());
    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_EQ((i % 2 != 0), result[i]);
    }
}

TEST(AllgatherTest, allgather_multiple_elements_kabool_with_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> input = {false, true};
    std::vector<kabool> result;
    comm.allgather(send_buf(input), recv_buf<resize_to_fit>(result));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), 2 * comm.size());
    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_EQ((i % 2 != 0), result[i]);
    }
}

TEST(AllgatherTest, allgather_default_container_type) {
    Communicator<OwnContainer> comm;
    size_t                     value = comm.rank();

    // This just has to compile
    OwnContainer<size_t> result = comm.allgather(send_buf(value));
}

TEST(AllgatherTest, send_recv_type_is_out_parameter) {
    Communicator           comm;
    std::vector<int> const data(1, comm.rank_signed());
    MPI_Datatype           send_type;
    MPI_Datatype           recv_type;
    auto recv_buf = comm.allgather(send_buf(data), send_type_out(send_type), recv_type_out(recv_type));

    EXPECT_EQ(send_type, MPI_INT);
    EXPECT_EQ(recv_type, MPI_INT);
    for (size_t i = 0; i < comm.size(); ++i) {
        EXPECT_EQ(recv_buf[i], i);
    }
}

TEST(AllgatherTest, send_recv_type_part_of_result_object) {
    Communicator           comm;
    std::vector<int> const data(1, comm.rank_signed());
    auto                   result = comm.allgather(send_buf(data), send_type_out(), recv_type_out());

    EXPECT_EQ(result.extract_send_type(), MPI_INT);
    EXPECT_EQ(result.extract_recv_type(), MPI_INT);
    auto recv_buf = result.extract_recv_buffer();
    for (size_t i = 0; i < comm.size(); ++i) {
        EXPECT_EQ(recv_buf[i], i);
    }
}

TEST(AllgatherTest, non_trivial_send_type) {
    // Each rank sends its rank two times to each other rank and receives the ranks without padding.
    Communicator     comm;
    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input{comm.rank_signed(), -1, -1, comm.rank_signed(), -1, -1};
    std::vector<int> recv_buffer(2 * comm.size(), 0);

    MPI_Type_commit(&int_padding_padding);
    auto res = comm.allgather(
        send_buf(input),
        send_type(int_padding_padding),
        send_count(2),
        recv_buf(recv_buffer),
        recv_count_out()
    );
    MPI_Type_free(&int_padding_padding);

    EXPECT_EQ(res.extract_recv_count(), 2);
    std::vector<int> expected_result;
    for (std::size_t i = 0; i < comm.size(); ++i) {
        expected_result.push_back(static_cast<int>(i));
        expected_result.push_back(static_cast<int>(i));
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AllgatherTest, non_trivial_recv_type) {
    // Each rank sends its rank two times (without padding) and receives the ranks with padding.
    Communicator     comm;
    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input{comm.rank_signed(), comm.rank_signed()};
    std::vector<int> recv_buffer(6 * comm.size(), -1);
    int              send_count_value = -1;

    MPI_Type_commit(&int_padding_padding);
    comm.allgather(
        send_buf(input),
        send_count_out(send_count_value),
        recv_buf(recv_buffer),
        recv_type(int_padding_padding),
        recv_count(2)
    );
    MPI_Type_free(&int_padding_padding);

    EXPECT_EQ(send_count_value, 2);
    std::vector<int> expected_result(6 * comm.size(), -1); // {0,-,-,0,-,-,1,-,-,1,-,-,...}
    for (std::size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 6]     = static_cast<int>(i);
        expected_result[i * 6 + 3] = static_cast<int>(i);
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AllgatherTest, different_send_and_recv_counts) {
    // Each rank sends its rank two times (without padding) and receives two ranks at a time.
    Communicator     comm;
    std::vector<int> input{comm.rank_signed(), comm.rank_signed()};
    std::vector<int> recv_buffer(3 * comm.size(), -1);
    MPI_Datatype     int_padding_int = MPI_INT_padding_MPI_INT();

    MPI_Type_commit(&int_padding_int);
    comm.allgather(send_buf(input), recv_buf(recv_buffer), recv_type(int_padding_int), recv_count(1));
    MPI_Type_free(&int_padding_int);

    std::vector<int> expected_result(3 * comm.size(), -1); // {0,-,0,1,-,1,...}
    for (std::size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 3]     = static_cast<int>(i);
        expected_result[i * 3 + 2] = static_cast<int>(i);
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AllgatherTest, different_send_and_recv_counts_without_explicit_mpi_types) {
    Communicator comm;
    struct CustomRecvStruct {
        int  a;
        int  b;
        bool operator==(CustomRecvStruct const& other) const {
            return std::tie(a, b) == std::tie(other.a, other.b);
        }
    };

    std::vector<int>              input{comm.rank_signed(), comm.rank_signed()};
    std::vector<CustomRecvStruct> recv_buffer(comm.size());
    comm.allgather(send_buf(input), recv_count(1), recv_buf(recv_buffer));

    std::vector<CustomRecvStruct> expected_result(comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i] = CustomRecvStruct{static_cast<int>(i), static_cast<int>(i)};
    }
    EXPECT_EQ(recv_buffer, expected_result);
}

TEST(AllgatherTest, structured_bindings) {
    Communicator           comm;
    std::vector<int>       input{comm.rank_signed()};
    std::vector<int> const expected_recv_buffer = [&]() {
        std::vector<int> vec(comm.size());
        std::iota(vec.begin(), vec.end(), 0);
        return vec;
    }();

    {
        // explicit recv buffer
        std::vector<int> recv_buffer(comm.size());
        auto [recv_count, send_count, recv_type, send_type] = comm.allgather(
            send_buf(input),
            recv_count_out(),
            recv_buf(recv_buffer),
            send_count_out(),
            recv_type_out(),
            send_type_out()
        );
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_count, 1);
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(recv_type, MPI_INT);
        EXPECT_EQ(send_type, MPI_INT);
    }
    {
        // implicit recv buffer
        auto [recv_buffer, recv_count, send_count, recv_type, send_type] =
            comm.allgather(send_buf(input), recv_count_out(), send_count_out(), recv_type_out(), send_type_out());
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_count, 1);
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(recv_type, MPI_INT);
        EXPECT_EQ(send_type, MPI_INT);
    }
    {
        // explicit but owning recv buffer
        auto [recv_count, send_count, recv_type, send_type, recv_buffer] = comm.allgather(
            send_buf(input),
            recv_count_out(),
            send_count_out(),
            recv_type_out(),
            send_type_out(),
            recv_buf(std::vector<int>(comm.size()))
        );
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_count, 1);
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(recv_type, MPI_INT);
        EXPECT_EQ(send_type, MPI_INT);
    }
    {
        // explicit but owning recv buffer and non-owning send_count
        int send_count                                       = -1;
        auto [recv_count, recv_type, send_type, recv_buffer] = comm.allgather(
            send_buf(input),
            recv_count_out(),
            send_count_out(send_count),
            recv_type_out(),
            send_type_out(),
            recv_buf(std::vector<int>(comm.size()))
        );
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_count, 1);
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(recv_type, MPI_INT);
        EXPECT_EQ(send_type, MPI_INT);
    }
    {
        // explicit but owning recv buffer and non-owning send_count, recv_type
        int          send_count = -1;
        MPI_Datatype recv_type;
        auto [recv_count, send_type, recv_buffer] = comm.allgather(
            send_buf(input),
            recv_count_out(),
            send_count_out(send_count),
            recv_type_out(recv_type),
            send_type_out(),
            recv_buf(std::vector<int>(comm.size()))
        );
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_count, 1);
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(recv_type, MPI_INT);
        EXPECT_EQ(send_type, MPI_INT);
    }
    {
        // explicit but owning recv buffer and non-owning send_count, recv_type (other order)
        int          send_count = -1;
        MPI_Datatype recv_type;
        auto [recv_count, send_type, recv_buffer] = comm.allgather(
            send_count_out(send_count),
            recv_type_out(recv_type),
            recv_count_out(),
            send_buf(input),
            send_type_out(),
            recv_buf(std::vector<int>(comm.size()))
        );
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_count, 1);
        EXPECT_EQ(send_count, 1);
        EXPECT_EQ(recv_type, MPI_INT);
        EXPECT_EQ(send_type, MPI_INT);
    }
}

TEST(AllgatherTest, inplace_basic) {
    Communicator     comm;
    std::vector<int> input(2 * comm.size(), -1);
    input[comm.rank() * 2]     = comm.rank_signed();
    input[comm.rank() * 2 + 1] = comm.rank_signed();
    comm.allgather(send_recv_buf(input));
    std::vector<int> expected_result(2 * comm.size(), -1);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    EXPECT_EQ(input, expected_result);
}

TEST(AllgatherTest, inplace_out_parameters) {
    Communicator     comm;
    std::vector<int> input(2 * comm.size(), -1);
    input[comm.rank() * 2]     = comm.rank_signed();
    input[comm.rank() * 2 + 1] = comm.rank_signed();
    auto [count, type]         = comm.allgather(send_recv_buf(input), send_recv_count_out(), send_recv_type_out());
    EXPECT_EQ(count, 2);
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(type));
    std::vector<int> expected_result(2 * comm.size(), -1);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    EXPECT_EQ(input, expected_result);
}

TEST(AllgatherTest, inplace_rvalue_buffer) {
    Communicator     comm;
    std::vector<int> input(2 * comm.size(), -1);
    input[comm.rank() * 2]     = comm.rank_signed();
    input[comm.rank() * 2 + 1] = comm.rank_signed();
    auto [output, count, type] =
        comm.allgather(send_recv_buf(std::move(input)), send_recv_count_out(), send_recv_type_out());
    EXPECT_EQ(count, 2);
    EXPECT_THAT(possible_mpi_datatypes<int>(), Contains(type));
    std::vector<int> expected_result(2 * comm.size(), -1);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    EXPECT_EQ(output, expected_result);
}

TEST(AllgatherTest, inplace_explicit_count) {
    Communicator comm;
    // make the buffer too big
    std::vector<int> input(2 * comm.size() + 5, -1);
    input[comm.rank() * 2]     = comm.rank_signed();
    input[comm.rank() * 2 + 1] = comm.rank_signed();
    comm.allgather(send_recv_buf(input), send_recv_count(2));
    std::vector<int> expected_result(2 * comm.size() + 5, -1);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    // the last 5 elements are untouched, because the buffer is not resized
    for (size_t i = 2 * comm.size(); i < expected_result.size(); ++i) {
        expected_result[i] = -1;
    }
    EXPECT_EQ(input, expected_result);
}

TEST(AllgatherTest, inplace_explicit_count_resize) {
    Communicator comm;
    // make the buffer too big
    std::vector<int> input(2 * comm.size() + 5, -1);
    input[comm.rank() * 2]     = comm.rank_signed();
    input[comm.rank() * 2 + 1] = comm.rank_signed();
    comm.allgather(send_recv_buf<resize_to_fit>(input), send_recv_count(2));
    std::vector<int> expected_result(2 * comm.size(), -1);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = static_cast<int>(i);
        expected_result[i * 2 + 1] = static_cast<int>(i);
    }
    EXPECT_EQ(input, expected_result);
}

TEST(AllgatherTest, inplace_explicit_type) {
    Communicator                     comm;
    std::vector<std::pair<int, int>> input(comm.size() * 2, std::make_pair(-1, -1));
    input[comm.rank() * 2]     = {comm.rank_signed(), comm.rank_signed() + 1};
    input[comm.rank() * 2 + 1] = {comm.rank_signed(), comm.rank_signed() + 1};
    MPI_Datatype type          = struct_type<std::pair<int, int>>::data_type();
    MPI_Type_commit(&type);
    comm.allgather(send_recv_buf(input), send_recv_type(type), send_recv_count(2));
    MPI_Type_free(&type);
    std::vector<std::pair<int, int>> expected_result(comm.size() * 2);
    for (size_t i = 0; i < comm.size(); ++i) {
        expected_result[i * 2]     = std::make_pair(static_cast<int>(i), static_cast<int>(i) + 1);
        expected_result[i * 2 + 1] = std::make_pair(static_cast<int>(i), static_cast<int>(i) + 1);
    }
    EXPECT_EQ(input, expected_result);
}
