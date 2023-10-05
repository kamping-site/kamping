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

#include "gmock/gmock.h"
#include <cstddef>
#include <numeric>

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

    auto result = comm.allgather(send_buf(value)).extract_recv_buffer();
    EXPECT_EQ(comm.root(), 0);
    ASSERT_EQ(result.size(), comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        EXPECT_EQ(result[i], i);
    }

    // Change default root and test with communicator's default root again, this should not change anything.
    comm.root(comm.size_signed() - 1);
    result = comm.allgather(send_buf(value)).extract_recv_buffer();
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

    auto result = comm.allgather(send_buf(value)).extract_recv_buffer();
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
    const std::vector<int> data(5, comm.rank_signed());
    int const              send_count = 1;
    int const              recv_count = 1;

    {
        // test that send_count parameter overwrites automatic deduction of send counts from the size of the send buffer
        auto result   = comm.allgather(send_buf(data), send_counts(send_count));
        auto recv_buf = result.extract_recv_buffer();
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(recv_buf[i], i);
        }
    }
    {
        // test that send_count parameter overwrites automatic deduction of send counts from the size of the send buffer
        auto result   = comm.allgather(send_buf(data), send_counts(send_count), recv_counts(recv_count));
        auto recv_buf = result.extract_recv_buffer();
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(recv_buf[i], i);
        }
    }
}

TEST(AllgatherTest, allgather_single_element_with_given_recv_buf_bigger_than_required) {
    Communicator           comm;
    const std::vector<int> data{comm.rank_signed()};
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
        expect_eq(Span(recv_buffer.data(), comm.size()), expected_recv_buffer);
    }
    {
        // recv buffer will not be resized as the policy is no_resize
        std::vector<int> recv_buffer(2 * comm.size());
        comm.allgather(send_buf(data), recv_buf<no_resize>(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        expect_eq(Span(recv_buffer.data(), comm.size()), expected_recv_buffer);
    }
    {
        // recv buffer will not be resized as the policy is no_resize (default)
        std::vector<int> recv_buffer(2 * comm.size());
        comm.allgather(send_buf(data), recv_buf(recv_buffer));
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        expect_eq(Span(recv_buffer.data(), comm.size()), expected_recv_buffer);
    }
}

TEST(AllgatherTest, given_recv_buffer_smaller_than_required) {
    Communicator           comm;
    const std::vector<int> data{comm.rank_signed()};
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
    auto             result = comm.allgather(send_buf(values)).extract_recv_buffer();

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
    auto result = comm.allgather(send_buf({false})).extract_recv_buffer();
    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), comm.size());
    for (auto elem: result) {
        EXPECT_EQ(elem, false);
    }
}

TEST(AllgatherTest, allgather_initializer_list_bool_no_receive_buffer) {
    Communicator comm;
    auto         result = comm.allgather(send_buf({false, false})).extract_recv_buffer();

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), 2 * comm.size());
    for (auto elem: result) {
        EXPECT_EQ(elem, false);
    }
}

TEST(AllgatherTest, allgather_single_element_kabool_no_receive_buffer) {
    Communicator comm;
    auto         result = comm.allgather(send_buf(kabool{false})).extract_recv_buffer();

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
    auto                result = comm.allgather(send_buf(input)).extract_recv_buffer();

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
    OwnContainer<size_t> result = comm.allgather(send_buf(value)).extract_recv_buffer();
}
