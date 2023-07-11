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

#include <cstddef>

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
    comm.allgatherv(send_buf(value), recv_buf(output));

    std::vector<decltype(value)> expected_output(comm.size());
    std::iota(expected_output.begin(), expected_output.end(), 0u);
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_and_allgather_have_same_result_for_single_element_receive_buffer) {
    Communicator comm;
    auto         value = comm.rank();

    std::vector<decltype(value)> output;
    std::vector<decltype(value)> output_v;

    comm.allgather(send_buf(value), recv_buf(output));
    comm.allgatherv(send_buf(value), recv_buf(output_v));
    EXPECT_EQ(output_v, output);
}

TEST(AllgathervTest, allgatherv_different_number_elems_to_send) {
    Communicator        comm;
    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()));
    std::vector<double> output;
    std::vector<double> expected_output =
        ExpectedBuffersForRankTimesRankGathering<>::recv_buffer_on_receiving_ranks<double>(comm);

    comm.allgatherv(send_buf(input), recv_buf(output));
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_different_number_elems_to_send_custom_container) {
    Communicator         comm;
    OwnContainer<double> input(comm.rank(), static_cast<double>(comm.rank()));
    std::vector<double>  output;
    std::vector<double>  expected_output =
        ExpectedBuffersForRankTimesRankGathering<>::recv_buffer_on_receiving_ranks<double>(comm);

    comm.allgatherv(send_buf(input), recv_buf(output));
    EXPECT_EQ(output, expected_output);
}

TEST(AllgathervTest, allgatherv_check_recv_counts_and_recv_displs) {
    Communicator     comm;
    std::vector<int> input(comm.rank(), static_cast<int>(comm.rank()));

    auto result = comm.allgatherv(send_buf(input));
    EXPECT_EQ(
        result.extract_recv_buffer(),
        ExpectedBuffersForRankTimesRankGathering<>::recv_buffer_on_receiving_ranks<int>(comm)
    );
    EXPECT_EQ(
        result.extract_recv_counts(),
        ExpectedBuffersForRankTimesRankGathering<>::recv_counts_on_receiving_ranks(comm)
    );
    EXPECT_EQ(
        result.extract_recv_displs(),
        ExpectedBuffersForRankTimesRankGathering<>::recv_displs_on_receiving_ranks(comm)
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
        ExpectedBuffersForRankTimesRankGathering<>::recv_buffer_on_receiving_ranks<int>(comm);

    // only provide recv_counts
    comm.allgatherv(send_buf(input), kamping::recv_counts(recv_counts), recv_buf(output));
    EXPECT_EQ(output, expected_output);

    // provide recv_counts and recv_displs
    comm.allgatherv(
        send_buf(input),
        kamping::recv_counts(recv_counts),
        kamping::recv_displs(recv_displs),
        recv_buf(output)
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
        kamping::recv_counts_out(recv_counts_out),
        recv_displs_out(alloc_new_auto<OwnContainer>)
    );
    EXPECT_EQ(result.extract_recv_buffer(), expected_output);
    EXPECT_EQ(recv_counts_out, expected_recv_counts);
    EXPECT_EQ(result.extract_recv_displs(), expected_recv_displs);
}
