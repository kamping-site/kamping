// This file is part of KaMPIng.
//
// Copyright 2021-2023 The KaMPIng Authors
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(GatherTest, gather_single_element_no_receive_buffer) {
    Communicator comm;
    auto         value = comm.rank();

    // Test default root of communicator
    auto result = comm.gather(send_buf(value));
    EXPECT_EQ(comm.root(), 0);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size_signed() - 1);
    result = comm.gather(send_buf(value));
    EXPECT_EQ(comm.root(), comm.size() - 1);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        result = comm.gather(send_buf(value), root(i));
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            ASSERT_EQ(result.size(), comm.size());
            for (size_t j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[j], j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, default_count_deduction) {
    Communicator     comm;
    std::vector<int> input(3, comm.rank_signed());
    { // send and recv count deduced
        int send_count = -1;
        int recv_count = -1;
        // send_count is deduced from send_buf.size()
        auto result = comm.gather(send_buf(input), send_count_out(send_count), recv_count_out(recv_count));
        EXPECT_EQ(send_count, 3);
        if (comm.is_root()) {
            EXPECT_EQ(recv_count, 3);
            EXPECT_EQ(result.size(), 3 * comm.size());
        } else {
            // left untouched on non-root ranks
            EXPECT_EQ(recv_count, -1);
        }
    }
    { // only recv count deduced
        int  recv_count = -1;
        auto result     = comm.gather(send_buf(input), send_count(1), recv_count_out(recv_count));
        if (comm.is_root()) {
            // recv count is deduced from send_buf size on root, untouched on other ranks
            EXPECT_EQ(recv_count, 1);
            EXPECT_EQ(result.size(), comm.size());
        } else {
            // left untouched on non-root ranks
            EXPECT_EQ(recv_count, -1);
        }
    }
}

TEST(GatherTest, send_recv_count_is_part_of_result_object) {
    Communicator     comm;
    std::vector<int> input(3, comm.rank_signed());
    { // send and recv count deduced
        // send_count is deduced from send_buf.size()
        auto result = comm.gather(send_buf(input), send_count_out(), recv_count_out());

        EXPECT_EQ(result.extract_send_count(), 3);
        if (comm.is_root()) {
            EXPECT_EQ(result.extract_recv_count(), 3);
            EXPECT_EQ(result.extract_recv_buffer().size(), 3 * comm.size());
        } else {
            // no assumption about content of recv count on non-root ranks
        }
    }
    { // only recv count deduced
        auto result = comm.gather(send_buf(input), send_count(1), recv_count_out());
        if (comm.is_root()) {
            // recv count is deduced from send_buf size on root, untouched on other ranks
            EXPECT_EQ(result.extract_recv_count(), 1);
            EXPECT_EQ(result.extract_recv_buffer().size(), comm.size());
        } else {
            // no assumption about content of recv count on non-root ranks
        }
    }
}

TEST(GatherTest, explicit_send_count_works) {
    Communicator     comm;
    std::vector<int> input(3, comm.rank_signed());
    auto             result = comm.gather(send_buf(input), send_count(1));
    if (comm.is_root()) {
        EXPECT_EQ(result.size(), comm.size());
        std::vector<int> expected_result(comm.size());
        std::iota(expected_result.begin(), expected_result.end(), 0);
        EXPECT_THAT(result, ElementsAreArray(expected_result));
    }
}

TEST(GatherTest, resize_policy_recv_buf_large_enough) {
    Communicator     comm;
    std::vector<int> output(comm.size() + 5, -1);
    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
    { // default resize policy (no resize)
        comm.gather(send_buf(comm.rank_signed()), recv_buf(output));
        if (comm.is_root()) {
            // buffer will not be resized
            EXPECT_EQ(output.size(), comm.size() + 5);
            EXPECT_THAT(Span(output.data(), comm.size()), ElementsAreArray(expected_result));
            EXPECT_THAT(Span(output.data() + comm.size(), output.size() - comm.size()), Each(Eq(-1)));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), comm.size() + 5);
            EXPECT_THAT(output, Each(Eq(-1)));
        }
    }
    { // no resize policy
        comm.gather(send_buf(comm.rank_signed()), recv_buf<no_resize>(output));
        if (comm.is_root()) {
            // buffer will not be resized
            EXPECT_EQ(output.size(), comm.size() + 5);
            EXPECT_THAT(Span(output.data(), comm.size()), ElementsAreArray(expected_result));
            EXPECT_THAT(Span(output.data() + comm.size(), output.size() - comm.size()), Each(Eq(-1)));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), comm.size() + 5);
            EXPECT_THAT(output, Each(Eq(-1)));
        }
    }
    { // grow only
        comm.gather(send_buf(comm.rank_signed()), recv_buf<grow_only>(output));
        if (comm.is_root()) {
            // buffer will not be resized
            EXPECT_EQ(output.size(), comm.size() + 5);
            EXPECT_THAT(Span(output.data(), comm.size()), ElementsAreArray(expected_result));
            EXPECT_THAT(Span(output.data() + comm.size(), output.size() - comm.size()), Each(Eq(-1)));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), comm.size() + 5);
            EXPECT_THAT(output, Each(Eq(-1)));
        }
    }
    { // resize to fit
        comm.gather(send_buf(comm.rank_signed()), recv_buf<resize_to_fit>(output));
        if (comm.is_root()) {
            // buffer will be resized
            EXPECT_THAT(output, ElementsAreArray(expected_result));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), comm.size() + 5);
            EXPECT_THAT(output, Each(Eq(-1)));
        }
    }
}

TEST(GatherTest, resize_policy_recv_buf_too_small) {
    Communicator     comm;
    std::vector<int> output(0);
    std::vector<int> expected_result(comm.size());
    std::iota(expected_result.begin(), expected_result.end(), 0);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
    { // default resize policy (no resize)
        if (comm.is_root()) {
            EXPECT_KASSERT_FAILS(
                comm.gather(send_buf(comm.rank_signed()), recv_buf(output)),
                "Recv buffer is not large enough to hold all received elements."
            );
            std::vector<int> large_enough_output(comm.size());
            // cleanup and do the actual gather
            int rank = comm.rank_signed();
            MPI_Gather(
                &rank,
                1,
                MPI_INT,
                large_enough_output.data(),
                1,
                MPI_INT,
                comm.root_signed(),
                comm.mpi_communicator()
            );
        } else {
            comm.gather(send_buf(comm.rank_signed()), recv_buf(output));
        }
    }
    { // no resize policy
        if (comm.is_root()) {
            EXPECT_KASSERT_FAILS(
                comm.gather(send_buf(comm.rank_signed()), recv_buf<no_resize>(output)),
                "Recv buffer is not large enough to hold all received elements."
            );
            std::vector<int> large_enough_output(comm.size());
            // cleanup and do the actual gather
            int rank = comm.rank_signed();
            MPI_Gather(
                &rank,
                1,
                MPI_INT,
                large_enough_output.data(),
                1,
                MPI_INT,
                comm.root_signed(),
                comm.mpi_communicator()
            );
        } else {
            comm.gather(send_buf(comm.rank_signed()), recv_buf(output));
        }
    }
#endif
    { // grow only
        comm.gather(send_buf(comm.rank_signed()), recv_buf<grow_only>(output));
        if (comm.is_root()) {
            // buffer will grow to fit
            EXPECT_THAT(output, ElementsAreArray(expected_result));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), 0);
        }
    }
    { // resize to fit
        comm.gather(send_buf(comm.rank_signed()), recv_buf<resize_to_fit>(output));
        if (comm.is_root()) {
            // buffer will be resized
            EXPECT_THAT(output, ElementsAreArray(expected_result));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), 0);
        }
    }
}

TEST(GatherTest, gather_single_custom_element_no_receive_buffer) {
    Communicator comm;
    struct CustomDataType {
        int rank;
        int additional_value;
    }; // struct custom_data_type

    CustomDataType value = {comm.rank_signed(), comm.size_signed() - comm.rank_signed()};

    // Test default root of communicator
    auto result = comm.gather(send_buf(value));
    EXPECT_EQ(comm.root(), 0);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i].rank, i);
            EXPECT_EQ(result[i].additional_value, comm.size() - i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size_signed() - 1);
    result = comm.gather(send_buf(value));
    EXPECT_EQ(comm.root(), comm.size() - 1);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i].rank, i);
            EXPECT_EQ(result[i].additional_value, comm.size() - i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        result = comm.gather(send_buf(value), root(i));
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            ASSERT_EQ(result.size(), comm.size());
            for (size_t j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[j].rank, j);
                EXPECT_EQ(result[j].additional_value, comm.size() - j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_single_element_with_receive_buffer) {
    Communicator                 comm;
    auto                         value = comm.rank();
    std::vector<decltype(value)> result(0);

    // Test default root of communicator
    comm.gather(send_buf(value), recv_buf<resize_to_fit>(result));
    EXPECT_EQ(comm.root(), 0);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size_signed() - 1);
    result.resize(0);
    comm.gather(send_buf(value), recv_buf<resize_to_fit>(result));
    EXPECT_EQ(comm.root(), comm.size() - 1);
    if (comm.rank() == comm.root()) {
        ASSERT_EQ(result.size(), comm.size());
        for (size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(result[i], i);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        result.resize(0);
        comm.gather(send_buf(value), recv_buf<resize_to_fit>(result), root(i));
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            ASSERT_EQ(result.size(), comm.size());
            for (size_t j = 0; j < comm.size(); ++j) {
                EXPECT_EQ(result[j], j);
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_multiple_elements_no_receive_buffer) {
    Communicator     comm;
    std::vector<int> values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    auto             result = comm.gather(send_buf(values));

    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    result = comm.gather(send_buf(values));
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        result = comm.gather(send_buf(values), root(i));
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            EXPECT_EQ(result.size(), values.size() * comm.size());
            for (size_t j = 0; j < result.size(); ++j) {
                EXPECT_EQ(result[j], j / values.size());
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_multiple_elements_with_receive_buffer) {
    Communicator     comm;
    std::vector<int> values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    std::vector<int> result(0);

    comm.gather(send_buf(values), recv_buf<resize_to_fit>(result));

    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Change default root and test with communicator's default root again
    comm.root(comm.size() - 1);
    result.resize(0);
    comm.gather(send_buf(values), recv_buf<resize_to_fit>(result));
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }

    // Pass any possible root to gather
    for (size_t i = 0; i < comm.size(); ++i) {
        result.resize(0);
        comm.gather(send_buf(values), root(i), recv_buf<resize_to_fit>(result));
        EXPECT_EQ(comm.root(), comm.size() - 1);
        if (comm.rank() == i) {
            EXPECT_EQ(result.size(), values.size() * comm.size());
            for (size_t j = 0; j < result.size(); ++j) {
                EXPECT_EQ(result[j], j / values.size());
            }
        } else {
            EXPECT_EQ(result.size(), 0);
        }
    }
}

TEST(GatherTest, gather_receive_custom_container) {
    Communicator      comm;
    std::vector<int>  values = {comm.rank_signed(), comm.rank_signed(), comm.rank_signed(), comm.rank_signed()};
    OwnContainer<int> result;

    comm.gather(send_buf(values), recv_buf<resize_to_fit>(result));

    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_send_custom_container) {
    Communicator      comm;
    OwnContainer<int> values(4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = comm.rank_signed();
    }
    std::vector<int> result;

    comm.gather(send_buf(values), recv_buf<resize_to_fit>(result));

    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_send_and_receive_custom_container) {
    Communicator      comm;
    OwnContainer<int> values(4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = comm.rank_signed();
    }
    OwnContainer<int> result;

    comm.gather(send_buf(values), recv_buf<resize_to_fit>(result));

    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), values.size() * comm.size());
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(result[i], i / values.size());
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_single_element_initializer_list_bool_no_receive_buffer) {
    Communicator comm;
    // gather does not support single element bool when specifying no recv_buffer, because the default receive buffer is
    // std::vector<bool>, which is not supported
    auto result = comm.gather(send_buf({false}));
    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), comm.size());
        for (auto elem: result) {
            EXPECT_EQ(elem, false);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_initializer_list_bool_no_receive_buffer) {
    Communicator comm;
    auto         result = comm.gather(send_buf({false, false}));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), 2 * comm.size());
        for (auto elem: result) {
            EXPECT_EQ(elem, false);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_single_element_kabool_no_receive_buffer) {
    Communicator comm;
    auto         result = comm.gather(send_buf(kabool{false}));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), comm.size());
        for (auto elem: result) {
            EXPECT_EQ(elem, false);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_single_element_bool_with_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> result;
    comm.gather(send_buf({false}), recv_buf<resize_to_fit>(result));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), comm.size());
        for (auto elem: result) {
            EXPECT_EQ(elem, false);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_single_element_kabool_with_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> result;
    comm.gather(send_buf(kabool{false}), recv_buf<resize_to_fit>(result));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), comm.size());
        for (auto elem: result) {
            EXPECT_EQ(elem, false);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_multiple_elements_kabool_no_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> input  = {false, true};
    auto                result = comm.gather(send_buf(input));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), 2 * comm.size());
        for (size_t i = 0; i < result.size(); i++) {
            EXPECT_EQ((i % 2 != 0), result[i]);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_multiple_elements_kabool_with_receive_buffer) {
    Communicator        comm;
    std::vector<kabool> input = {false, true};
    std::vector<kabool> result;
    comm.gather(send_buf(input), recv_buf<resize_to_fit>(result));

    KASSERT((std::is_same_v<decltype(result), std::vector<kabool>>));
    // Test default root of communicator
    if (comm.rank() == comm.root()) {
        EXPECT_EQ(result.size(), 2 * comm.size());
        for (size_t i = 0; i < result.size(); i++) {
            EXPECT_EQ((i % 2 != 0), result[i]);
        }
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(GatherTest, gather_default_container_type) {
    Communicator<OwnContainer> comm;
    size_t                     value = comm.rank();

    // This just has to compile
    OwnContainer<size_t> result = comm.gather(send_buf(value));
}

TEST(GatherTest, gather_send_recv_type_are_out_parameters) {
    Communicator comm;

    MPI_Datatype     send_type = MPI_CHAR;
    MPI_Datatype     recv_type = MPI_CHAR;
    std::vector<int> result;
    comm.gather(
        send_buf(comm.rank_signed()),
        recv_buf<resize_to_fit>(result),
        send_type_out(send_type),
        recv_type_out(recv_type)
    );

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), comm.size());
    } else {
        EXPECT_EQ(result.size(), 0);
    }
    EXPECT_EQ(send_type, MPI_INT);
    EXPECT_EQ(recv_type, MPI_INT);
}

TEST(GatherTest, gather_send_recv_type_are_part_of_result_object) {
    Communicator comm;

    std::vector<int> result;
    auto             res =
        comm.gather(send_buf(comm.rank_signed()), recv_buf<resize_to_fit>(result), send_type_out(), recv_type_out());

    if (comm.is_root()) {
        EXPECT_EQ(result.size(), comm.size());
    } else {
        EXPECT_EQ(result.size(), 0);
    }
    EXPECT_EQ(res.extract_send_type(), MPI_INT);
    EXPECT_EQ(res.extract_recv_type(), MPI_INT);
}

TEST(GatherTest, non_trivial_send_type) {
    // each rank sends its rank two times with padding and the root rank receives the messages without
    // padding.
    Communicator     comm;
    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input;
    int const        root_rank = comm.size_signed() / 2;
    std::vector<int> recv_buffer;
    if (comm.is_root(root_rank)) {
        recv_buffer.resize(2 * comm.size());
    }

    MPI_Type_commit(&int_padding_padding);
    auto res = comm.gather(
        root(root_rank),
        send_buf({comm.rank_signed(), -1, -1, comm.rank_signed(), -1, -1}),
        send_type(int_padding_padding),
        send_count(2),
        recv_buf(recv_buffer),
        recv_count_out()
    );
    MPI_Type_free(&int_padding_padding);

    if (comm.is_root(root_rank)) {
        EXPECT_EQ(res.extract_recv_count(), 2);
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        for (int i = 0; i < comm.size_signed(); ++i) {
            EXPECT_EQ(recv_buffer[static_cast<size_t>(2 * i)], i);
            EXPECT_EQ(recv_buffer[static_cast<size_t>(2 * i + 1)], i);
        }
    } else {
        EXPECT_EQ(recv_buffer.size(), 0);
    }
}

TEST(GatherTest, non_trivial_recv_type) {
    // each rank sends its rank two times without padding and the root rank receives the messages with
    // padding.
    Communicator     comm;
    MPI_Datatype     int_padding_padding = MPI_INT_padding_padding();
    std::vector<int> input;
    int const        root_rank = comm.size_signed() / 2;
    std::vector<int> recv_buffer;
    if (comm.is_root(root_rank)) {
        recv_buffer.resize(3 * 2 * comm.size());
    }

    MPI_Type_commit(&int_padding_padding);
    auto res = comm.gather(
        root(root_rank),
        send_buf({comm.rank_signed(), comm.rank_signed()}),
        send_count_out(),
        recv_type(int_padding_padding),
        recv_count(2),
        recv_buf(recv_buffer)
    );
    MPI_Type_free(&int_padding_padding);

    EXPECT_EQ(res.extract_send_count(), 2);
    if (comm.is_root(root_rank)) {
        EXPECT_EQ(recv_buffer.size(), 3 * 2 * comm.size());
        for (int i = 0; i < comm.size_signed(); ++i) {
            EXPECT_EQ(recv_buffer[static_cast<size_t>(6 * i)], i);
            EXPECT_EQ(recv_buffer[static_cast<size_t>(6 * i + 3)], i);
        }
    } else {
        EXPECT_EQ(recv_buffer.size(), 0);
    }
}

TEST(GatherTest, different_send_and_recv_counts) {
    // each rank sends its rank two times and the root rank receives the two messages at once (with padding in the
    // middle).
    Communicator     comm;
    MPI_Datatype     int_padding_int = MPI_INT_padding_MPI_INT();
    std::vector<int> recv_buffer;
    if (comm.is_root()) {
        recv_buffer.resize(3 * comm.size());
    }
    int send_count = -1;

    MPI_Type_commit(&int_padding_int);
    comm.gather(
        send_buf({comm.rank_signed(), comm.rank_signed()}),
        send_count_out(send_count),
        recv_buf(recv_buffer),
        recv_type(int_padding_int),
        recv_count(1)
    );
    MPI_Type_free(&int_padding_int);

    EXPECT_EQ(send_count, 2);
    if (comm.is_root()) {
        EXPECT_EQ(send_count, 2);
        EXPECT_EQ(recv_buffer.size(), 3 * comm.size());
        for (std::size_t i = 0; i < comm.size(); ++i) {
            EXPECT_EQ(recv_buffer[3 * i], static_cast<int>(i));
            EXPECT_EQ(recv_buffer[3 * i + 2], static_cast<int>(i));
        }
    } else {
        EXPECT_EQ(recv_buffer.size(), 0);
    }
}

struct CustomRecvStruct {
    int  a;
    int  b;
    bool operator==(CustomRecvStruct const& other) const {
        return std::tie(a, b) == std::tie(other.a, other.b);
    }
    friend std::ostream& operator<<(std::ostream& out, CustomRecvStruct const& str) {
        return out << "(" << str.a << ", " << str.b << ")";
    }
};

TEST(GatherTest, different_send_and_recv_counts_without_explicit_mpi_types) {
    Communicator comm;

    std::vector<CustomRecvStruct> recv_buffer;
    if (comm.is_root()) {
        recv_buffer.resize(comm.size());
    }
    int send_count = -1;

    comm.gather(
        send_buf({comm.rank_signed(), comm.rank_signed()}),
        send_count_out(send_count),
        recv_count(1),
        recv_buf(recv_buffer)
    );

    EXPECT_EQ(send_count, 2);
    if (comm.is_root()) {
        EXPECT_EQ(recv_buffer.size(), comm.size());
        for (int i = 0; i < comm.size_signed(); ++i) {
            CustomRecvStruct expected_elem{i, i};
            EXPECT_EQ(recv_buffer[static_cast<size_t>(i)], expected_elem);
        }
    } else {
        EXPECT_EQ(recv_buffer.size(), 0);
    }
}
// Death test do not work with MPI.
/// @todo Implement proper tests for input validation via KASSERT()s.
// TEST(GatherTest, gather_different_roots_on_different_processes) {
//     Communicator comm;
//     auto         value = comm.rank();
//
//     if (kassert::internal::assertion_enabled(assert::light_communication) && comm.size() > 1) {
//         EXPECT_KASSERT_FAILS(comm.gather(send_buf(value), root(comm.rank())), "Root has to be the same on all
//         ranks.");
//     }
// }
//

TEST(GatherTest, structured_bindings) {
    Communicator           comm;
    std::vector<int>       input{comm.rank_signed()};
    std::vector<int> const expected_recv_buffer_on_root = [&]() {
        std::vector<int> vec(comm.size());
        std::iota(vec.begin(), vec.end(), 0);
        return vec;
    }();

    {
        // explicit recv buffer
        std::vector<int> recv_buffer(comm.size());
        auto [recv_count, send_count, recv_type, send_type] = comm.gather(
            send_buf(input),
            recv_count_out(),
            recv_buf(recv_buffer),
            send_count_out(),
            recv_type_out(),
            send_type_out()
        );
        if (comm.is_root()) {
            EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
            EXPECT_EQ(recv_type, MPI_INT);
            EXPECT_EQ(recv_count, 1);
            EXPECT_EQ(send_count, 1);
        } else {
            EXPECT_EQ(recv_buffer, std::vector<int>(comm.size()));
            EXPECT_EQ(send_type, MPI_INT);
            EXPECT_EQ(recv_count, 0);
            EXPECT_EQ(send_count, 1);
        }
    }
    {
        // implicit recv buffer
        auto [recv_buffer, recv_count, send_count, recv_type, send_type] =
            comm.gather(send_buf(input), recv_count_out(), send_count_out(), recv_type_out(), send_type_out());
        if (comm.is_root()) {
            EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
            EXPECT_EQ(recv_count, 1);
            EXPECT_EQ(send_count, 1);
            EXPECT_EQ(send_type, MPI_INT);
        } else {
            EXPECT_EQ(recv_buffer.size(), 0);
            EXPECT_EQ(send_count, 1);
            EXPECT_EQ(recv_count, 0);
            EXPECT_EQ(recv_type, MPI_INT);
        }
    }
    {
        // explicit but owning recv buffer
        auto [recv_count, send_count, recv_type, send_type, recv_buffer] = comm.gather(
            send_buf(input),
            recv_count_out(),
            send_count_out(),
            recv_type_out(),
            send_type_out(),
            recv_buf(std::vector<int>(comm.size()))
        );
        if (comm.is_root()) {
            EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
            EXPECT_EQ(recv_type, MPI_INT);
            EXPECT_EQ(recv_count, 1);
            EXPECT_EQ(send_count, 1);
        } else {
            EXPECT_EQ(recv_buffer, std::vector<int>(comm.size()));
            EXPECT_EQ(send_type, MPI_INT);
            EXPECT_EQ(send_count, 1);
            EXPECT_EQ(recv_count, 0);
        }
    }
    {
        // explicit but owning recv buffer and non-owning send_count
        int send_count                                       = -1;
        auto [recv_count, recv_type, send_type, recv_buffer] = comm.gather(
            send_buf(input),
            recv_count_out(),
            send_count_out(send_count),
            recv_type_out(),
            send_type_out(),
            recv_buf(std::vector<int>(comm.size()))
        );
        if (comm.is_root()) {
            EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
            EXPECT_EQ(recv_count, 1);
            EXPECT_EQ(recv_type, MPI_INT);
            EXPECT_EQ(send_count, 1);
        } else {
            EXPECT_EQ(recv_buffer, std::vector<int>(comm.size()));
            EXPECT_EQ(send_count, 1);
            EXPECT_EQ(send_type, MPI_INT);
            EXPECT_EQ(recv_count, 0);
        }
    }
    {
        // explicit but owning recv buffer and non-owning send_count, recv_type
        int          send_count = -1;
        MPI_Datatype recv_type;
        auto [recv_count, send_type, recv_buffer] = comm.gather(
            send_buf(input),
            recv_count_out(),
            send_count_out(send_count),
            recv_type_out(recv_type),
            send_type_out(),
            recv_buf(std::vector<int>(comm.size()))
        );
        if (comm.is_root()) {
            EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
            EXPECT_EQ(recv_count, 1);
            EXPECT_EQ(recv_type, MPI_INT);
            EXPECT_EQ(send_count, 1);
        } else {
            EXPECT_EQ(recv_buffer, std::vector<int>(comm.size()));
            EXPECT_EQ(send_count, 1);
            EXPECT_EQ(send_type, MPI_INT);
            EXPECT_EQ(recv_count, 0);
        }
    }
    {
        // explicit but owning recv buffer and non-owning send_count, recv_type (other order) and root parameter
        int          send_count = -1;
        int          root       = comm.size_signed() - 1;
        MPI_Datatype recv_type;
        auto [recv_count, send_type, recv_buffer] = comm.gather(
            send_count_out(send_count),
            recv_type_out(recv_type),
            recv_count_out(),
            send_buf(input),
            send_type_out(),
            recv_buf(std::vector<int>(comm.size())),
            kamping::root(root)
        );
        if (comm.is_root(root)) {
            EXPECT_EQ(recv_buffer, expected_recv_buffer_on_root);
            EXPECT_EQ(recv_count, 1);
            EXPECT_EQ(recv_type, MPI_INT);
            EXPECT_EQ(send_count, 1);
        } else {
            EXPECT_EQ(recv_buffer, std::vector<int>(comm.size()));
            EXPECT_EQ(send_count, 1);
            EXPECT_EQ(send_type, MPI_INT);
            EXPECT_EQ(recv_count, 0);
        }
    }
}
