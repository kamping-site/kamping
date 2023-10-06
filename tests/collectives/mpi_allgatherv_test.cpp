// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "gatherv_test_helpers.hpp"
#include "kamping/collectives/allgather.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(AllgathervTest, allgatherv_single_element_no_receive_buffer) {
    Communicator comm;
    auto         value  = comm.rank();
    auto         output = comm.allgatherv(send_buf(value)).extract_recv_buffer();

    std::vector<decltype(value)> expected_output(comm.size());
    std::iota(expected_output.begin(), expected_output.end(), 0u);
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_single_element_explicit_send_count) {
    Communicator        comm;
    std::vector<size_t> data(5, comm.rank());
    int const           send_count = 1;
    // although send_buffer contains 5 elements only one element is gathered (as set via the explicit send_count
    // parameter)
    auto output = comm.allgatherv(send_buf(data), send_counts(send_count)).extract_recv_buffer();

    std::vector<size_t> expected_output(comm.size());
    std::iota(expected_output.begin(), expected_output.end(), 0u);
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_and_allgather_have_same_result_for_single_element_no_receive_buffer) {
    Communicator comm;
    auto         value = comm.rank();

    auto output   = comm.allgather(send_buf(value)).extract_recv_buffer();
    auto output_v = comm.allgatherv(send_buf(value)).extract_recv_buffer();
    EXPECT_EQ(output_v, output);
}

TEST(AllgathervTest, allgatherv_single_element_receive_buffer) {
    Communicator                 comm;
    auto                         value = comm.rank();
    std::vector<decltype(value)> output;
    comm.allgatherv(send_buf(value), recv_buf<resize_to_fit>(output));

    std::vector<decltype(value)> expected_output(comm.size());
    std::iota(expected_output.begin(), expected_output.end(), 0u);
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_and_allgather_have_same_result_for_single_element_receive_buffer) {
    Communicator comm;
    auto         value = comm.rank();

    std::vector<decltype(value)> output;
    std::vector<decltype(value)> output_v;

    comm.allgather(send_buf(value), recv_buf<resize_to_fit>(output));
    comm.allgatherv(send_buf(value), recv_buf<resize_to_fit>(output_v));
    EXPECT_EQ(output_v, output);
}

TEST(AllgathervTest, allgatherv_different_number_elems_to_send) {
    Communicator        comm;
    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()));
    std::vector<double> output;
    std::vector<double> expected_output =
        ExpectedBuffersForRankTimesRankGathering::recv_buffer_on_receiving_ranks<double>(comm);

    comm.allgatherv(send_buf(input), recv_buf<resize_to_fit>(output));
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_different_number_elems_to_send_custom_container) {
    Communicator         comm;
    OwnContainer<double> input(comm.rank(), static_cast<double>(comm.rank()));
    std::vector<double>  output;
    std::vector<double>  expected_output =
        ExpectedBuffersForRankTimesRankGathering::recv_buffer_on_receiving_ranks<double>(comm);

    comm.allgatherv(send_buf(input), recv_buf<resize_to_fit>(output));
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_check_recv_counts_and_recv_displs) {
    Communicator     comm;
    std::vector<int> input(comm.rank(), static_cast<int>(comm.rank()));

    auto result = comm.allgatherv(send_buf(input));
    EXPECT_EQ(
        result.extract_recv_buffer(),
        ExpectedBuffersForRankTimesRankGathering::recv_buffer_on_receiving_ranks<int>(comm)
    );
    EXPECT_EQ(
        result.extract_recv_counts(),
        ExpectedBuffersForRankTimesRankGathering::recv_counts_on_receiving_ranks(comm)
    );
    EXPECT_EQ(
        result.extract_recv_displs(),
        ExpectedBuffersForRankTimesRankGathering::recv_displs_on_receiving_ranks(comm)
    );
}

TEST(AllgathervTest, allgatherv_provide_recv_counts_and_recv_displs) {
    Communicator     comm;
    std::vector<int> input(comm.rank(), static_cast<int>(comm.rank()));
    std::vector<int> recv_counts(comm.size());
    std::vector<int> recv_displs(comm.size());
    std::iota(recv_counts.begin(), recv_counts.end(), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);

    std::vector<int> output;
    std::vector<int> expected_output =
        ExpectedBuffersForRankTimesRankGathering::recv_buffer_on_receiving_ranks<int>(comm);

    // only provide recv_counts
    comm.allgatherv(send_buf(input), kamping::recv_counts(recv_counts), recv_buf<resize_to_fit>(output));
    EXPECT_EQ(output, expected_output);

    // provide recv_counts and recv_displs
    comm.allgatherv(
        send_buf(input),
        kamping::recv_counts(recv_counts),
        kamping::recv_displs(recv_displs),
        recv_buf<resize_to_fit>(output)
    );
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_all_empty_but_rank_in_the_middle) {
    Communicator     comm;
    size_t           non_empty_rank = comm.size() / 2;
    std::vector<int> input;
    if (comm.rank() == non_empty_rank) {
        input.resize(comm.rank(), comm.rank_signed());
    }
    std::vector<int> expected_output(non_empty_rank, static_cast<int>(non_empty_rank));
    std::vector<int> expected_recv_counts(comm.size(), 0);
    std::vector<int> expected_recv_displs(comm.size());
    expected_recv_counts[non_empty_rank] = static_cast<int>(non_empty_rank);
    std::exclusive_scan(expected_recv_counts.begin(), expected_recv_counts.end(), expected_recv_displs.begin(), 0);

    auto result = comm.allgatherv(send_buf(input));
    EXPECT_EQ(result.extract_recv_buffer(), expected_output);
    EXPECT_EQ(result.extract_recv_counts(), expected_recv_counts);
    EXPECT_EQ(result.extract_recv_displs(), expected_recv_displs);
}

TEST(AllgathervTest, allgatherv_all_empty_but_rank_in_the_middle_with_different_container_types) {
    Communicator     comm;
    size_t           non_empty_rank = comm.size() / 2;
    std::vector<int> input;
    if (comm.rank() == non_empty_rank) {
        input.resize(comm.rank(), comm.rank_signed());
    }
    OwnContainer<int> recv_counts_out;
    std::vector<int>  expected_output(non_empty_rank, static_cast<int>(non_empty_rank));
    OwnContainer<int> expected_recv_counts(comm.size(), 0);
    OwnContainer<int> expected_recv_displs(comm.size());

    expected_recv_counts[non_empty_rank] = static_cast<int>(non_empty_rank);
    std::exclusive_scan(expected_recv_counts.begin(), expected_recv_counts.end(), expected_recv_displs.begin(), 0);

    auto result = comm.allgatherv(
        send_buf(input),
        kamping::recv_counts_out<resize_to_fit>(recv_counts_out),
        recv_displs_out(alloc_new_auto<OwnContainer>)
    );
    EXPECT_EQ(result.extract_recv_buffer(), expected_output);
    EXPECT_EQ(recv_counts_out, expected_recv_counts);
    EXPECT_EQ(result.extract_recv_displs(), expected_recv_displs);
}

TEST(AllgathervTest, allgather_single_element_with_given_buffers_bigger_than_required) {
    Communicator           comm;
    const std::vector<int> data{comm.rank_signed()};
    std::vector<int>       expected_recv_buffer(comm.size());
    std::iota(expected_recv_buffer.begin(), expected_recv_buffer.end(), 0);
    std::vector<int> expected_recv_counts(comm.size(), 1);
    std::vector<int> expected_recv_displs(comm.size());
    std::iota(expected_recv_displs.begin(), expected_recv_displs.end(), 0);

    {
        // recv buffer will be resized to the size of the communicator
        std::vector<int> recv_buffer(2 * comm.size());
        std::vector<int> recv_counts(2 * comm.size());
        std::vector<int> recv_displs(2 * comm.size());
        comm.allgatherv(
            send_buf(data),
            recv_counts_out<resize_to_fit>(recv_counts),
            recv_displs_out<resize_to_fit>(recv_displs),
            recv_buf<resize_to_fit>(recv_buffer)
        );
        EXPECT_EQ(recv_buffer.size(), comm.size());
        EXPECT_EQ(recv_counts.size(), comm.size());
        EXPECT_EQ(recv_displs.size(), comm.size());
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_counts, expected_recv_counts);
        EXPECT_EQ(recv_displs, expected_recv_displs);
    }
    {
        // recv buffer will not be resized as it is large enough and policy is grow_only
        std::vector<int> recv_buffer(2 * comm.size());
        std::vector<int> recv_counts(2 * comm.size());
        std::vector<int> recv_displs(2 * comm.size());
        comm.allgatherv(
            send_buf(data),
            recv_counts_out<grow_only>(recv_counts),
            recv_displs_out<grow_only>(recv_displs),
            recv_buf<grow_only>(recv_buffer)
        );
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_counts.size(), 2 * comm.size());
        EXPECT_EQ(recv_displs.size(), 2 * comm.size());
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
        EXPECT_THAT(Span(recv_counts.data(), comm.size()), ElementsAreArray(expected_recv_counts));
        EXPECT_THAT(Span(recv_displs.data(), comm.size()), ElementsAreArray(expected_recv_displs));
    }
    {
        // recv buffer will not be resized as the policy is no_resize
        std::vector<int> recv_buffer(2 * comm.size());
        std::vector<int> recv_counts(2 * comm.size());
        std::vector<int> recv_displs(2 * comm.size());
        comm.allgatherv(
            send_buf(data),
            recv_counts_out<no_resize>(recv_counts),
            recv_displs_out<no_resize>(recv_displs),
            recv_buf<no_resize>(recv_buffer)
        );
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_counts.size(), 2 * comm.size());
        EXPECT_EQ(recv_displs.size(), 2 * comm.size());
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
        EXPECT_THAT(Span(recv_counts.data(), comm.size()), ElementsAreArray(expected_recv_counts));
        EXPECT_THAT(Span(recv_displs.data(), comm.size()), ElementsAreArray(expected_recv_displs));
    }
    {
        // recv buffer will not be resized as the policy is no_resize (default)
        std::vector<int> recv_buffer(2 * comm.size());
        std::vector<int> recv_counts(2 * comm.size());
        std::vector<int> recv_displs(2 * comm.size());
        comm.allgatherv(
            send_buf(data),
            recv_counts_out(recv_counts),
            recv_displs_out(recv_displs),
            recv_buf(recv_buffer)
        );
        EXPECT_EQ(recv_buffer.size(), 2 * comm.size());
        EXPECT_EQ(recv_counts.size(), 2 * comm.size());
        EXPECT_EQ(recv_displs.size(), 2 * comm.size());
        EXPECT_THAT(Span(recv_buffer.data(), comm.size()), ElementsAreArray(expected_recv_buffer));
        EXPECT_THAT(Span(recv_counts.data(), comm.size()), ElementsAreArray(expected_recv_counts));
        EXPECT_THAT(Span(recv_displs.data(), comm.size()), ElementsAreArray(expected_recv_displs));
    }
}

TEST(AllgathervTest, allgather_single_element_with_given_buffers_smaller_than_required) {
    Communicator           comm;
    const std::vector<int> data{comm.rank_signed()};
    std::vector<int>       expected_recv_buffer(comm.size());
    std::iota(expected_recv_buffer.begin(), expected_recv_buffer.end(), 0);
    std::vector<int> expected_recv_counts(comm.size(), 1);
    std::vector<int> expected_recv_displs(comm.size());
    std::iota(expected_recv_displs.begin(), expected_recv_displs.end(), 0);

    {
        // recv buffer will be resized to the size of the communicator
        std::vector<int> recv_buffer(comm.size() / 2);
        std::vector<int> recv_counts(comm.size() / 2);
        std::vector<int> recv_displs(comm.size() / 2);
        comm.allgatherv(
            send_buf(data),
            recv_counts_out<resize_to_fit>(recv_counts),
            recv_displs_out<resize_to_fit>(recv_displs),
            recv_buf<resize_to_fit>(recv_buffer)
        );
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_counts, expected_recv_counts);
        EXPECT_EQ(recv_displs, expected_recv_displs);
    }
    {
        // recv buffer will be resized as it is not large enough and policy is grow_only
        std::vector<int> recv_buffer(comm.size() / 2);
        std::vector<int> recv_counts(comm.size() / 2);
        std::vector<int> recv_displs(comm.size() / 2);
        comm.allgatherv(
            send_buf(data),
            recv_counts_out<grow_only>(recv_counts),
            recv_displs_out<grow_only>(recv_displs),
            recv_buf<grow_only>(recv_buffer)
        );
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
        EXPECT_EQ(recv_counts, expected_recv_counts);
        EXPECT_EQ(recv_displs, expected_recv_displs);
    }
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
TEST(AllgatherTest, given_buffers_smaller_than_required_with_policy_no_resize) {
    Communicator comm;

    std::vector<int> input{comm.rank_signed()};
    {
        // test kassert for sufficient size of recv_counts
        std::vector<int> recv_buffer;
        std::vector<int> recv_counts;
        std::vector<int> recv_displs;
        EXPECT_KASSERT_FAILS(
            comm.allgatherv(
                send_buf(input),
                recv_counts_out<no_resize>(recv_counts),
                recv_displs_out<resize_to_fit>(recv_displs),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
        // test kassert for sufficient size of recv_counts (default is no_resize)
        EXPECT_KASSERT_FAILS(
            comm.allgatherv(
                send_buf(input),
                recv_counts_out(recv_counts),
                recv_displs_out<resize_to_fit>(recv_displs),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
    }
    {
        // test kassert for sufficient size of recv_displs
        std::vector<int> recv_buffer;
        std::vector<int> recv_counts;
        std::vector<int> recv_displs;
        EXPECT_KASSERT_FAILS(
            comm.allgatherv(
                send_buf(input),
                recv_counts_out<resize_to_fit>(recv_counts),
                recv_displs_out<no_resize>(recv_displs),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
        // test kassert for sufficient size of recv_displs (default is no_resize)
        EXPECT_KASSERT_FAILS(
            comm.allgatherv(
                send_buf(input),
                recv_counts_out<resize_to_fit>(recv_counts),
                recv_displs_out(recv_displs),
                recv_buf<resize_to_fit>(recv_buffer)
            ),
            ""
        );
    }
    {
        // test kassert for sufficient size of recv_counts
        std::vector<int> recv_buffer;
        std::vector<int> recv_counts;
        std::vector<int> recv_displs;
        EXPECT_KASSERT_FAILS(
            comm.allgatherv(
                send_buf(input),
                recv_counts_out<resize_to_fit>(recv_counts),
                recv_displs_out<resize_to_fit>(recv_displs),
                recv_buf<no_resize>(recv_buffer)
            ),
            ""
        );
        // test kassert for sufficient size of recv_counts (default is no_resize)
        EXPECT_KASSERT_FAILS(
            comm.allgatherv(
                send_buf(input),
                recv_counts_out<resize_to_fit>(recv_counts),
                recv_displs_out<resize_to_fit>(recv_displs),
                recv_buf(recv_buffer)
            ),
            ""
        );
    }
}
#endif

TEST(AllgathervTest, non_monotonically_increasing_recv_displacements) {
    // Rank i sends its rank i times. The messages are received in reverse order via
    // explicit recv_displs. E.g. for 4 PES we expect that recv_buf contains [3,3,3,2,2,1]
    Communicator comm;

    // prepare send buffer
    std::vector<int> input(comm.rank(), comm.rank_signed());

    // prepare recv counts and displs
    std::vector<int> recv_counts(comm.size());
    std::iota(recv_counts.begin(), recv_counts.end(), 0);
    std::vector<int> recv_displs(comm.size());
    std::exclusive_scan(recv_counts.rbegin(), recv_counts.rend(), recv_displs.begin(), 0u);
    std::reverse(recv_displs.begin(), recv_displs.end());

    auto expected_recv_buffer = [&]() {
        std::vector<int> expected_recv_buf;
        for (int i = 0; i < comm.size_signed(); ++i) {
            int source_rank = comm.size_signed() - 1 - i;
            std::fill_n(std::back_inserter(expected_recv_buf), source_rank, source_rank);
        }
        return expected_recv_buf;
    };

    {
        // do the allgatherv without recv_counts
        auto recv_buf = comm.allgatherv(send_buf(input), kamping::recv_displs(recv_displs)).extract_recv_buffer();
        EXPECT_EQ(recv_buf, expected_recv_buffer());
    }
    {
        // do the allgatherv with recv_counts
        auto recv_buf =
            comm.allgatherv(send_buf(input), kamping::recv_counts(recv_counts), kamping::recv_displs(recv_displs))
                .extract_recv_buffer();
        EXPECT_EQ(recv_buf, expected_recv_buffer());
    }
}
