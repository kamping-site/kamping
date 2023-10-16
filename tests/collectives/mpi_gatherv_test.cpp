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

#include <gmock/gmock.h>
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

TEST(GathervTest, resize_policy_all_buffers_are_large_enough) {
    Communicator     comm;
    std::vector<int> input(comm.rank(), comm.rank_signed());
    std::vector<int> expected_recv_counts(comm.size());
    std::iota(expected_recv_counts.begin(), expected_recv_counts.end(), 0);
    std::vector<int> expected_recv_displs(comm.size());
    std::exclusive_scan(expected_recv_counts.begin(), expected_recv_counts.end(), expected_recv_displs.begin(), 0);
    std::vector<int> expected_result;
    for (int i = 0; i < comm.size_signed(); i++) {
        for (int j = 0; j < i; j++) {
            expected_result.push_back(i);
        }
    }
    { // default resize policy (no resize)
        std::vector<int> output(expected_result.size() + 5, -1);
        std::vector<int> recv_counts_output(comm.size() + 5, -1);
        std::vector<int> recv_displs_output(comm.size() + 5, -1);
        comm.gatherv(
            send_buf(input),
            recv_buf(output),
            recv_counts_out(recv_counts_output),
            recv_displs_out(recv_displs_output)
        );
        EXPECT_EQ(output.size(), expected_result.size() + 5);
        EXPECT_EQ(recv_counts_output.size(), expected_recv_counts.size() + 5);
        EXPECT_EQ(recv_displs_output.size(), expected_recv_displs.size() + 5);
        if (comm.is_root()) {
            EXPECT_THAT(Span(output.data(), expected_result.size()), ElementsAreArray(expected_result));
            EXPECT_THAT(Span(output.data() + expected_result.size(), 5), Each(Eq(-1)));

            EXPECT_THAT(
                Span(recv_counts_output.data(), expected_recv_counts.size()),
                ElementsAreArray(expected_recv_counts)
            );
            EXPECT_THAT(Span(recv_counts_output.data() + expected_recv_counts.size(), 5), Each(Eq(-1)));

            EXPECT_THAT(
                Span(recv_displs_output.data(), expected_recv_displs.size()),
                ElementsAreArray(expected_recv_displs)
            );
            EXPECT_THAT(Span(recv_displs_output.data() + expected_recv_displs.size(), 5), Each(Eq(-1)));
        } else {
            // buffer will not be touched
            EXPECT_THAT(output, Each(Eq(-1)));
            EXPECT_THAT(recv_counts_output, Each(Eq(-1)));
            EXPECT_THAT(recv_displs_output, Each(Eq(-1)));
        }
    }
    { // no resize policy
        std::vector<int> output(expected_result.size() + 5, -1);
        std::vector<int> recv_counts_output(comm.size() + 5, -1);
        std::vector<int> recv_displs_output(comm.size() + 5, -1);
        comm.gatherv(
            send_buf(input),
            recv_buf<no_resize>(output),
            recv_counts_out<no_resize>(recv_counts_output),
            recv_displs_out<no_resize>(recv_displs_output)
        );
        EXPECT_EQ(output.size(), expected_result.size() + 5);
        EXPECT_EQ(recv_counts_output.size(), expected_recv_counts.size() + 5);
        EXPECT_EQ(recv_displs_output.size(), expected_recv_displs.size() + 5);
        if (comm.is_root()) {
            EXPECT_THAT(Span(output.data(), expected_result.size()), ElementsAreArray(expected_result));
            EXPECT_THAT(Span(output.data() + expected_result.size(), 5), Each(Eq(-1)));

            EXPECT_THAT(
                Span(recv_counts_output.data(), expected_recv_counts.size()),
                ElementsAreArray(expected_recv_counts)
            );
            EXPECT_THAT(Span(recv_counts_output.data() + expected_recv_counts.size(), 5), Each(Eq(-1)));

            EXPECT_THAT(
                Span(recv_displs_output.data(), expected_recv_displs.size()),
                ElementsAreArray(expected_recv_displs)
            );
            EXPECT_THAT(Span(recv_displs_output.data() + expected_recv_displs.size(), 5), Each(Eq(-1)));
        } else {
            // buffer will not be touched
            EXPECT_THAT(output, Each(Eq(-1)));
            EXPECT_THAT(recv_counts_output, Each(Eq(-1)));
            EXPECT_THAT(recv_displs_output, Each(Eq(-1)));
        }
    }
    { // grow only
        std::vector<int> output(expected_result.size() + 5, -1);
        std::vector<int> recv_counts_output(comm.size() + 5, -1);
        std::vector<int> recv_displs_output(comm.size() + 5, -1);
        comm.gatherv(
            send_buf(input),
            recv_buf<grow_only>(output),
            recv_counts_out<grow_only>(recv_counts_output),
            recv_displs_out<grow_only>(recv_displs_output)
        );
        EXPECT_EQ(output.size(), expected_result.size() + 5);
        EXPECT_EQ(recv_counts_output.size(), expected_recv_counts.size() + 5);
        EXPECT_EQ(recv_displs_output.size(), expected_recv_displs.size() + 5);
        if (comm.is_root()) {
            EXPECT_THAT(Span(output.data(), expected_result.size()), ElementsAreArray(expected_result));
            EXPECT_THAT(Span(output.data() + expected_result.size(), 5), Each(Eq(-1)));

            EXPECT_THAT(
                Span(recv_counts_output.data(), expected_recv_counts.size()),
                ElementsAreArray(expected_recv_counts)
            );
            EXPECT_THAT(Span(recv_counts_output.data() + expected_recv_counts.size(), 5), Each(Eq(-1)));

            EXPECT_THAT(
                Span(recv_displs_output.data(), expected_recv_displs.size()),
                ElementsAreArray(expected_recv_displs)
            );
            EXPECT_THAT(Span(recv_displs_output.data() + expected_recv_displs.size(), 5), Each(Eq(-1)));
        } else {
            // buffer will not be touched
            EXPECT_THAT(output, Each(Eq(-1)));
            EXPECT_THAT(recv_counts_output, Each(Eq(-1)));
            EXPECT_THAT(recv_displs_output, Each(Eq(-1)));
        }
    }
    { // resize to fit
        std::vector<int> output(expected_result.size() + 5, -1);
        std::vector<int> recv_counts_output(comm.size() + 5, -1);
        std::vector<int> recv_displs_output(comm.size() + 5, -1);
        comm.gatherv(
            send_buf(input),
            recv_buf<resize_to_fit>(output),
            recv_counts_out<resize_to_fit>(recv_counts_output),
            recv_displs_out<resize_to_fit>(recv_displs_output)
        );
        if (comm.is_root()) {
            EXPECT_THAT(output, ElementsAreArray(expected_result));
            EXPECT_THAT(recv_counts_output, ElementsAreArray(expected_recv_counts));
            EXPECT_THAT(recv_displs_output, ElementsAreArray(expected_recv_displs));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), expected_result.size() + 5);
            EXPECT_EQ(recv_counts_output.size(), expected_recv_counts.size() + 5);
            EXPECT_EQ(recv_displs_output.size(), expected_recv_displs.size() + 5);
            EXPECT_THAT(output, Each(Eq(-1)));
            EXPECT_THAT(recv_counts_output, Each(Eq(-1)));
            EXPECT_THAT(recv_displs_output, Each(Eq(-1)));
        }
    }
}

TEST(GathervTest, resize_policy_all_buffers_are_too_small) {
    Communicator     comm;
    std::vector<int> input(comm.rank(), comm.rank_signed());
    std::vector<int> expected_recv_counts(comm.size());
    std::iota(expected_recv_counts.begin(), expected_recv_counts.end(), 0);
    std::vector<int> expected_recv_displs(comm.size());
    std::exclusive_scan(expected_recv_counts.begin(), expected_recv_counts.end(), expected_recv_displs.begin(), 0);
    std::vector<int> expected_result;
    for (int i = 0; i < comm.size_signed(); i++) {
        for (int j = 0; j < i; j++) {
            expected_result.push_back(i);
        }
    }

    // @todo tests for failed assertions in case of resize policy no_resize are
    // ommitted here, because they would require excessive cleanup of dangling
    // collective communication.

    { // grow only
        std::vector<int> output(0);
        std::vector<int> recv_counts_output(0);
        std::vector<int> recv_displs_output(0);
        comm.gatherv(
            send_buf(input),
            recv_buf<grow_only>(output),
            recv_counts_out<grow_only>(recv_counts_output),
            recv_displs_out<grow_only>(recv_displs_output)
        );
        if (comm.is_root()) {
            EXPECT_THAT(output, ElementsAreArray(expected_result));
            EXPECT_THAT(recv_counts_output, ElementsAreArray(expected_recv_counts));
            EXPECT_THAT(recv_displs_output, ElementsAreArray(expected_recv_displs));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), 0);
            EXPECT_EQ(recv_counts_output.size(), 0);
            EXPECT_EQ(recv_displs_output.size(), 0);
        }
    }
    { // resize to fit
        std::vector<int> output(0);
        std::vector<int> recv_counts_output(0);
        std::vector<int> recv_displs_output(0);
        comm.gatherv(
            send_buf(input),
            recv_buf<resize_to_fit>(output),
            recv_counts_out<resize_to_fit>(recv_counts_output),
            recv_displs_out<resize_to_fit>(recv_displs_output)
        );
        if (comm.is_root()) {
            EXPECT_THAT(output, ElementsAreArray(expected_result));
            EXPECT_THAT(recv_counts_output, ElementsAreArray(expected_recv_counts));
            EXPECT_THAT(recv_displs_output, ElementsAreArray(expected_recv_displs));
        } else {
            // buffer will not be touched
            EXPECT_EQ(output.size(), 0);
            EXPECT_EQ(recv_counts_output.size(), 0);
            EXPECT_EQ(recv_displs_output.size(), 0);
        }
    }
}
TEST(GathervTest, recv_counts_ignore_on_non_root_works) {
    Communicator comm;
    if (comm.is_root()) {
        auto computed_recv_counts = comm.gatherv(send_buf(comm.rank_signed())).extract_recv_counts();
        EXPECT_EQ(computed_recv_counts.size(), comm.size());
        EXPECT_THAT(computed_recv_counts, Each(Eq(1)));
    } else {
        comm.gatherv(send_buf(comm.rank_signed()), recv_counts(ignore<>));
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
