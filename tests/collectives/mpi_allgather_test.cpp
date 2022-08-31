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

#include "gmock/gmock.h"
#include <cstddef>

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

    comm.allgather(send_buf(value), recv_buf(result));
    ASSERT_EQ(result.size(), comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        EXPECT_EQ(result[i], i);
    }
}

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

    comm.allgather(send_buf(values), recv_buf(result));

    EXPECT_EQ(result.size(), values.size() * comm.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], i / values.size());
    }
}

TEST(AllgatherTest, allgather_receive_custom_container) {
    Communicator      comm;
    std::vector<int>  values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    OwnContainer<int> result;

    comm.allgather(send_buf(values), recv_buf(result));

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

    comm.allgather(send_buf(values), recv_buf(result));

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

    comm.allgather(send_buf(values), recv_buf(result));

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
    comm.allgather(send_buf({false}), recv_buf(result));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), comm.size());
    for (auto elem: result) {
        EXPECT_EQ(elem, false);
    }
}

TEST(AllgatherTest, allgather_single_element_kabool_with_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> result;
    comm.allgather(send_buf(kabool{false}), recv_buf(result));

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
    comm.allgather(send_buf(input), recv_buf(result));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    EXPECT_EQ(result.size(), 2 * comm.size());
    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_EQ((i % 2 != 0), result[i]);
    }
}
