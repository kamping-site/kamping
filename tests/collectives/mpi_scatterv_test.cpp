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

#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

namespace {
std::vector<int>
create_equiv_sized_input_vector_on_root(Communicator const& comm, int const elements_per_rank, int root = -1) {
    if (root < 0) {
        root = comm.root_signed();
    }

    std::vector<int> input;
    if (comm.rank_signed() == root) {
        input.resize(static_cast<std::size_t>(elements_per_rank) * comm.size());
        for (int rank = 0; rank < comm.size_signed(); ++rank) {
            auto begin = input.begin() + rank * elements_per_rank;
            auto end   = begin + elements_per_rank;
            std::fill(begin, end, rank);
        }
    }
    return input;
}

std::vector<int> create_equiv_counts_on_root(Communicator const& comm, int const elements_per_rank, int root = -1) {
    if (root < 0) {
        root = comm.root_signed();
    }

    std::vector<int> counts;
    if (comm.rank_signed() == root) {
        counts.resize(comm.size());
        std::fill(counts.begin(), counts.end(), elements_per_rank);
    }
    return counts;
}
} // namespace

TEST(ScattervTest, scatterv_equiv_single_element_return_recv_buf) {
    Communicator comm;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const counts = create_equiv_counts_on_root(comm, 1);
    auto const result = comm.scatterv(send_buf(input), send_counts(counts), recv_counts(1)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_recv_buf) {
    Communicator comm;

    auto const       input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const       counts = create_equiv_counts_on_root(comm, 1);
    std::vector<int> result;
    comm.scatterv(send_buf(input), send_counts(counts), recv_counts(1), recv_buf(result));

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_recv_buf_var) {
    Communicator comm;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const counts = create_equiv_counts_on_root(comm, 1);
    int        result;
    comm.scatterv(send_buf(input), send_counts(counts), recv_counts(1), recv_buf(result));

    EXPECT_EQ(result, comm.rank());
}

TEST(ScattervTest, scatterv_equiv_single_element_no_recv_counts) {
    Communicator comm;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const counts = create_equiv_counts_on_root(comm, 1);
    auto const result = comm.scatterv(send_buf(input), send_counts(counts)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_recv_counts) {
    Communicator comm;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const counts = create_equiv_counts_on_root(comm, 1);
    int        recv_count;
    int        result;
    comm.scatterv(send_buf(input), send_counts(counts), recv_counts_out(recv_count), recv_buf(result));

    EXPECT_EQ(recv_count, 1);
    EXPECT_EQ(result, comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_send_counts) {
    Communicator comm;

    auto const       input      = create_equiv_sized_input_vector_on_root(comm, 1);
    int const        recv_count = 1;
    int              result;
    std::vector<int> send_counts;
    comm.scatterv(send_buf(input), send_counts_out(send_counts), recv_counts(recv_count), recv_buf(result));

    if (comm.is_root()) {
        EXPECT_EQ(send_counts.size(), comm.size());
        EXPECT_THAT(send_counts, Each(Eq(1)));
    }
    EXPECT_EQ(result, comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_single_element_return_send_counts) {
    Communicator comm;

    auto const input      = create_equiv_sized_input_vector_on_root(comm, 1);
    int const  recv_count = 1;
    int        result;
    auto send_counts = comm.scatterv(send_buf(input), recv_counts(recv_count), recv_buf(result)).extract_send_counts();

    if (comm.is_root()) {
        EXPECT_EQ(send_counts.size(), comm.size());
        EXPECT_THAT(send_counts, Each(Eq(1)));
    }
    EXPECT_EQ(result, comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_single_element_out_send_displs) {
    Communicator comm;

    auto const       input  = create_equiv_sized_input_vector_on_root(comm, 1);
    auto const       counts = create_equiv_counts_on_root(comm, 1);
    int              result;
    std::vector<int> displs;
    comm.scatterv(send_buf(input), send_counts(counts), send_displs_out(displs), recv_buf(result));

    if (comm.is_root()) {
        EXPECT_EQ(displs.size(), comm.size());
        for (int pe = 0; pe < comm.size_signed(); ++pe) {
            EXPECT_EQ(displs[static_cast<std::size_t>(pe)], pe);
        }
    }

    EXPECT_EQ(result, comm.rank_signed());
}

TEST(ScattervTest, scatterv_equiv_multiple_elements) {
    Communicator comm;

    auto const       input  = create_equiv_sized_input_vector_on_root(comm, comm.size_signed());
    auto const       counts = create_equiv_counts_on_root(comm, comm.size_signed());
    std::vector<int> displs;
    int              recv_count;
    auto const       result =
        comm.scatterv(send_buf(input), send_counts(counts), send_displs_out(displs), recv_counts_out(recv_count))
            .extract_recv_buffer();

    if (comm.is_root()) {
        ASSERT_EQ(displs.size(), comm.size());
        for (int pe = 0; pe < comm.size_signed(); ++pe) {
            EXPECT_EQ(displs[static_cast<std::size_t>(pe)], pe * comm.size_signed());
        }
    }

    EXPECT_EQ(recv_count, comm.size_signed());
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(Eq(comm.rank_signed())));
}

TEST(ScattervTest, scatterv_nonequiv) {
    Communicator comm;

    // Send i+1 elements to PE i
    std::vector<int> input;
    std::vector<int> counts;
    for (int pe = 0; pe < comm.size_signed(); ++pe) {
        for (int i = 0; i <= pe; ++i) {
            input.push_back(pe);
        }
        counts.push_back(pe + 1);
    }

    int        recv_count;
    auto const result =
        comm.scatterv(send_buf(input), send_counts(counts), recv_counts_out(recv_count)).extract_recv_buffer();

    EXPECT_EQ(recv_count, comm.rank_signed() + 1);
    EXPECT_EQ(result.size(), comm.rank() + 1);
    EXPECT_THAT(result, Each(Eq(comm.rank_signed())));
}

TEST(ScattervTest, scatterv_nonzero_root) {
    Communicator comm;
    int const    root_val = comm.size_signed() - 1;

    auto const input  = create_equiv_sized_input_vector_on_root(comm, 1, root_val);
    auto const counts = create_equiv_counts_on_root(comm, 1, root_val);

    auto const result =
        comm.scatterv(send_buf(input), root(root_val), send_counts(counts), recv_counts(1)).extract_recv_buffer();

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result.front(), comm.rank_signed());
}
