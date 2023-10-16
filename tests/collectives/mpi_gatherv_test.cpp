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
#include <numeric>

#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "gatherv_test_helpers.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(GathervTest, gather_single_element_on_different_roots) {
    Communicator comm;
    auto         value = comm.rank();

    auto test_result = [&](auto&& mpi_result, int root) {
        if (comm.rank_signed() == root) {
            std::vector<int> expected_recv_counts(comm.size(), 1);
            std::vector<int> expected_recv_displs;
            std::exclusive_scan(
                expected_recv_counts.begin(),
                expected_recv_counts.end(),
                std::back_inserter(expected_recv_displs),
                0
            );
            std::vector<decltype(value)> expected_output(comm.size());
            std::iota(expected_output.begin(), expected_output.end(), 0);
            EXPECT_EQ(mpi_result.extract_recv_buffer(), expected_output);
            EXPECT_EQ(mpi_result.extract_recv_counts(), expected_recv_counts);
            EXPECT_EQ(mpi_result.extract_recv_displs(), expected_recv_displs);
        } else {
            // out recv buffers on non-root ranks are expected to be empty
            EXPECT_EQ(mpi_result.extract_recv_buffer().size(), 0u);
        }
    };

    // test with communicator's default root
    {
        EXPECT_EQ(comm.root(), 0);
        auto result = comm.gatherv(send_buf(value));
        test_result(result, 0);
    }
    // test with communicator's default root
    {
        int const new_default_root = comm.size_signed() - 1;
        comm.root(new_default_root);
        auto result = comm.gatherv(send_buf(value));
        test_result(result, new_default_root);
    }
    // test with all other possible roots
    {
        for (int i = 1; i + 1 < comm.size_signed(); ++i) {
            auto result = comm.gatherv(send_buf(value), root(i));
            test_result(result, i);
        }
    }
}

TEST(GathervTest, gather_varying_number_elements_on_different_roots) {
    Communicator        comm;
    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()));

    auto test_result = [&](auto&& mpi_result, int root) {
        if (comm.rank_signed() == root) {
            EXPECT_EQ(
                mpi_result.extract_recv_buffer(),
                ExpectedBuffersForRankTimesRankGathering::recv_buffer_on_receiving_ranks<double>(comm)
            );
            EXPECT_EQ(
                mpi_result.extract_recv_counts(),
                ExpectedBuffersForRankTimesRankGathering::recv_counts_on_receiving_ranks(comm)
            );
            EXPECT_EQ(
                mpi_result.extract_recv_displs(),
                ExpectedBuffersForRankTimesRankGathering::recv_displs_on_receiving_ranks(comm)
            );
        } else {
            // out recv buffers on non-root ranks are expected to be empty
            EXPECT_EQ(mpi_result.extract_recv_buffer().size(), 0u);
        }
    };

    // test with communicator's default root
    {
        EXPECT_EQ(comm.root(), 0);
        auto result = comm.gatherv(send_buf(input));
        test_result(result, 0);
    }
    // test with communicator's default root
    {
        int const new_default_root = comm.size_signed() - 1;
        comm.root(new_default_root);
        auto result = comm.gatherv(send_buf(input));
        test_result(result, new_default_root);
    }
    // test with all other possible roots
    {
        for (int i = 1; i + 1 < comm.size_signed(); ++i) {
            auto result = comm.gatherv(send_buf(input), root(i));
            test_result(result, i);
        }
    }
}

TEST(GathervTest, gather_varying_number_elements_on_different_roots_with_explicit_recv_counts_and_displacements) {
    Communicator        comm;
    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()));

    auto test_result = [&](auto&& mpi_result, int root) {
        if (comm.rank_signed() == root) {
            EXPECT_EQ(
                mpi_result.extract_recv_buffer(),
                ExpectedBuffersForRankTimesRankGathering::recv_buffer_on_receiving_ranks<double>(comm)
            );
        } else {
            // out recv buffers on non-root ranks are expected to be empty
            EXPECT_EQ(mpi_result.extract_recv_buffer().size(), 0u);
        }
    };

    // test with all possible roots
    {
        for (int i = 0; i < comm.size_signed(); ++i) {
            std::vector<int> recv_counts =
                ExpectedBuffersForRankTimesRankGathering::recv_counts_on_receiving_ranks(comm);
            std::vector<int> recv_displs =
                ExpectedBuffersForRankTimesRankGathering::recv_displs_on_receiving_ranks(comm);
            if (!comm.is_root(i)) {
                // invalid input for non root ranks as these should ignore recv counts/displacement buffers
                std::fill(recv_counts.begin(), recv_counts.end(), -1);
                std::fill(recv_displs.begin(), recv_displs.end(), -1);
            }
            auto result = comm.gatherv(
                send_buf(input),
                root(i),
                kamping::recv_counts(recv_counts),
                kamping::recv_displs(recv_displs)
            );
            test_result(result, i);
        }
    }
}

TEST(GathervTest, gather_mix_different_container_types) {
    Communicator         comm;
    OwnContainer<double> input(comm.rank(), static_cast<double>(comm.rank()));

    // test with all possible roots
    {
        for (int i = 0; i < comm.size_signed(); ++i) {
            std::vector<int> recv_counts =
                ExpectedBuffersForRankTimesRankGathering::recv_counts_on_receiving_ranks(comm);
            if (!comm.is_root(i)) {
                // invalid input for non root ranks as these should ignore recv counts/displacement buffers
                recv_counts.clear();
            }
            auto mpi_result = comm.gatherv(
                send_buf(input),
                root(i),
                recv_buf(alloc_new<std::vector<double>>),
                kamping::recv_counts(recv_counts),
                kamping::recv_displs_out(alloc_new_auto<OwnContainer>)
            );

            if (comm.rank_signed() == i) {
                EXPECT_EQ(
                    mpi_result.extract_recv_buffer(),
                    ExpectedBuffersForRankTimesRankGathering::recv_buffer_on_receiving_ranks<double>(comm)
                );
                EXPECT_EQ(
                    mpi_result.extract_recv_displs(),
                    ExpectedBuffersForRankTimesRankGathering::recv_displs_on_receiving_ranks<OwnContainer>(comm)
                );
            } else {
                // out recv buffers on non-root ranks are expected to be empty
                EXPECT_EQ(mpi_result.extract_recv_buffer().size(), 0u);
            }
        }
    }
}

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_LIGHT)
TEST(GathervTest, recv_counts_ignore_should_fail_on_root) {
    Communicator comm;
    if (comm.is_root()) {
        EXPECT_KASSERT_FAILS(
            comm.gatherv(send_buf(comm.rank_signed()), recv_counts(ignore<>)),
            "Recv counts buffer is smaller than the number of PEs at the root PE."
        )
        // cleanup
        comm.gatherv(send_buf(comm.rank_signed()));
    } else {
        comm.gatherv(send_buf(comm.rank_signed()), recv_counts(ignore<>));
    }
}
#endif
