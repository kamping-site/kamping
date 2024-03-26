// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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

#include "kamping/plugin/alltoall_grid.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::plugin;

using namespace grid_plugin_helpers;

TEST(AlltoallvGridTest, alltoallv_single_element) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto result = grid_comm.alltoallv(send_buf(input), kamping::send_counts(send_counts));
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(comm.rank_signed()));
}

TEST(AlltoallvGridTest, alltoallv_single_element_get_send_displs) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto [result, send_displs] =
        grid_comm.alltoallv(send_buf(input), kamping::send_counts(send_counts), send_displs_out());
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(comm.rank_signed()));

    std::vector<int> expected_send_displs(comm.size());
    std::iota(expected_send_displs.begin(), expected_send_displs.end(), 0);
    EXPECT_EQ(send_displs, expected_send_displs);
}

TEST(AlltoallvGridTest, alltoallv_single_element_get_recv_displs) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto [result, recv_displs] =
        grid_comm.alltoallv(send_buf(input), kamping::send_counts(send_counts), recv_displs_out());
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(comm.rank_signed()));

    std::vector<int> expected_recv_displs(comm.size());
    std::iota(expected_recv_displs.begin(), expected_recv_displs.end(), 0);
    EXPECT_EQ(recv_displs, expected_recv_displs);
}

TEST(AlltoallvGridTest, alltoallv_single_element_provide_send_displs) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    size_t const     send_displs_offset = 10;
    std::vector<int> input(send_displs_offset + comm.size());
    std::iota(input.begin() + send_displs_offset, input.end(), 0);
    std::vector<int> send_displs(comm.size());
    std::iota(send_displs.begin(), send_displs.end(), send_displs_offset);
    std::vector<int> send_counts(comm.size(), 1);
    auto             result =
        grid_comm.alltoallv(send_buf(input), kamping::send_counts(send_counts), kamping::send_displs(send_displs));
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(Span(result.begin(), result.end()), Each(comm.rank_signed()));
}

TEST(AlltoallvGridTest, alltoallv_single_element_provide_recv_displs) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    size_t const     recv_displ_offset = 10;
    std::vector<int> recv_displs(comm.size());
    std::iota(recv_displs.begin(), recv_displs.end(), recv_displ_offset);
    std::vector<int> send_counts(comm.size(), 1);

    auto result =
        grid_comm.alltoallv(send_buf(input), kamping::send_counts(send_counts), kamping::recv_displs(recv_displs));
    EXPECT_EQ(result.size(), comm.size() + recv_displ_offset);
    EXPECT_THAT(Span(result.begin(), result.begin() + recv_displ_offset), Each(0));
    EXPECT_THAT(Span(result.begin() + recv_displ_offset, result.end()), Each(comm.rank_signed()));
}

TEST(AlltoallvGridTest, alltoallv_single_element_st_binding) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto [recv_buf, recv_counts] =
        grid_comm.alltoallv(recv_counts_out(), send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(recv_buf.size(), comm.size());
    EXPECT_THAT(recv_buf, Each(comm.rank_signed()));
    EXPECT_EQ(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, Each(1));
}

TEST(AlltoallvGridTest, alltoallv_single_element_st_binding_recv_buf_provided) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);
    std::vector<int> recv_buf(comm.size());

    auto [recv_counts] = grid_comm.alltoallv(
        recv_counts_out(),
        kamping::recv_buf(recv_buf),
        send_buf(input),
        kamping::send_counts(send_counts)
    );

    EXPECT_EQ(recv_buf.size(), comm.size());
    EXPECT_THAT(recv_buf, Each(comm.rank_signed()));
    EXPECT_EQ(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, Each(1));
}

TEST(AlltoallvGridTest, alltoallv_single_element_st_binding_recv_buf_provided_resize) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);
    std::vector<int> recv_buf;

    auto [recv_counts] = grid_comm.alltoallv(
        recv_counts_out(),
        kamping::recv_buf<resize_to_fit>(recv_buf),
        send_buf(input),
        kamping::send_counts(send_counts)
    );

    EXPECT_EQ(recv_buf.size(), comm.size());
    EXPECT_THAT(recv_buf, Each(comm.rank_signed()));
    EXPECT_EQ(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, Each(1));
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_single_element_no_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto result = grid_comm.alltoallv_with_envelope(send_buf(input), kamping::send_counts(send_counts));
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(comm.rank_signed()));
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_single_element_source_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<double> input(comm.size());
    std::iota(input.begin(), input.end(), 0.5);
    std::vector<int> send_counts(comm.size(), 1);

    constexpr auto envelope = MessageEnvelopeLevel::source;

    auto result = grid_comm.alltoallv_with_envelope<envelope>(send_buf(input), kamping::send_counts(send_counts));
    EXPECT_EQ(result.size(), comm.size());

    for (size_t i = 0; i < comm.size(); ++i) {
        auto it = std::find_if(result.begin(), result.end(), [&](auto const& msg) { return msg.get_source() == i; });
        EXPECT_NE(it, result.end());
        EXPECT_EQ(it->get_payload(), static_cast<double>(comm.rank()) + 0.5);
    }
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_single_element_source_destination_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<double> input(comm.size());
    std::iota(input.begin(), input.end(), 0.5);
    std::vector<int> send_counts(comm.size(), 1);

    constexpr auto envelope = MessageEnvelopeLevel::source_and_destination;
    auto result = grid_comm.alltoallv_with_envelope<envelope>(send_buf(input), kamping::send_counts(send_counts));
    EXPECT_EQ(result.size(), comm.size());

    for (size_t i = 0; i < comm.size(); ++i) {
        auto it = std::find_if(result.begin(), result.end(), [&](auto const& msg) { return msg.get_source() == i; });
        EXPECT_NE(it, result.end());
        EXPECT_EQ(it->get_payload(), static_cast<double>(comm.rank()) + 0.5);
        EXPECT_EQ(it->get_destination(), comm.rank());
    }
}

TEST(AlltoallvGridTest, alltoallv_last_to_all_pe) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    int                 last_pe = comm.size_signed() - 1;
    std::vector<double> input;
    std::vector<int>    send_counts(comm.size(), 0);
    if (comm.is_root(last_pe)) {
        int const count = comm.size_signed() * (comm.size_signed() - 1) / 2;
        input.resize(static_cast<size_t>(count), static_cast<double>(last_pe) + 0.5);
        std::iota(send_counts.begin(), send_counts.end(), 0);
    }

    auto [recv_buf, recv_counts] =
        grid_comm.alltoallv(recv_counts_out(), send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(recv_buf.size(), comm.rank());
    EXPECT_THAT(recv_buf, Each(last_pe + 0.5));
    std::vector<int> expected_recv_counts(comm.size(), 0);
    expected_recv_counts.back() = comm.rank_signed();
    EXPECT_EQ(recv_counts, expected_recv_counts);
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_last_to_all_pe_no_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    int                 last_pe = comm.size_signed() - 1;
    std::vector<double> input;
    std::vector<int>    send_counts(comm.size(), 0);
    if (comm.is_root(last_pe)) {
        int const count = comm.size_signed() * (comm.size_signed() - 1) / 2;
        input.resize(static_cast<size_t>(count), static_cast<double>(last_pe) + 0.5);
        std::iota(send_counts.begin(), send_counts.end(), 0);
    }

    auto result = grid_comm.alltoallv_with_envelope(send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(result.size(), comm.rank());
    EXPECT_THAT(result, Each(last_pe + 0.5));
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_last_to_all_pe_source_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    int                 last_pe = comm.size_signed() - 1;
    std::vector<double> input;
    std::vector<int>    send_counts(comm.size(), 0);
    if (comm.is_root(last_pe)) {
        int const count = comm.size_signed() * (comm.size_signed() - 1) / 2;
        input.resize(static_cast<size_t>(count), static_cast<double>(last_pe) + 0.5);
        std::iota(send_counts.begin(), send_counts.end(), 0);
    }

    constexpr auto envelope = MessageEnvelopeLevel::source;
    auto result = grid_comm.alltoallv_with_envelope<envelope>(send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(result.size(), comm.rank());
    for (auto const& elem: result) {
        EXPECT_EQ(elem.get_payload(), static_cast<double>(last_pe) + 0.5);
        EXPECT_EQ(elem.get_source(), last_pe);
    }
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_last_to_all_pe_source_destination_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    int                 last_pe = comm.size_signed() - 1;
    std::vector<double> input;
    std::vector<int>    send_counts(comm.size(), 0);
    if (comm.is_root(last_pe)) {
        int const count = comm.size_signed() * (comm.size_signed() - 1) / 2;
        input.resize(static_cast<size_t>(count), static_cast<double>(last_pe) + 0.5);
        std::iota(send_counts.begin(), send_counts.end(), 0);
    }

    constexpr auto envelope = MessageEnvelopeLevel::source_and_destination;
    auto result = grid_comm.alltoallv_with_envelope<envelope>(send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(result.size(), comm.rank());
    for (auto const& elem: result) {
        EXPECT_EQ(elem.get_payload(), static_cast<double>(last_pe) + 0.5);
        EXPECT_EQ(elem.get_source(), last_pe);
        EXPECT_EQ(elem.get_destination(), comm.rank());
    }
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_all_to_last_pe_no_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()) + 0.5);
    std::vector<int>    send_counts(comm.size(), 0);
    send_counts[comm.size() - 1] = comm.rank_signed();

    auto result = grid_comm.alltoallv_with_envelope(send_buf(input), kamping::send_counts(send_counts));

    if (comm.is_root(comm.size() - 1)) {
        EXPECT_EQ(result.size(), comm.size() * (comm.size() - 1) / 2);
        std::vector<int> expected_recv_counts(comm.size());
        std::iota(expected_recv_counts.begin(), expected_recv_counts.end(), 0);
        for (auto const& elem: result) {
            size_t source = static_cast<size_t>(elem - 0.5);
            --expected_recv_counts[source];
        }
        EXPECT_THAT(expected_recv_counts, Each(0));
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_all_to_last_pe_source_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()) + 0.5);
    std::vector<int>    send_counts(comm.size(), 0);
    send_counts[comm.size() - 1] = comm.rank_signed();

    constexpr auto envelope = MessageEnvelopeLevel::source;
    auto result = grid_comm.alltoallv_with_envelope<envelope>(send_buf(input), kamping::send_counts(send_counts));

    if (comm.is_root(comm.size() - 1)) {
        EXPECT_EQ(result.size(), comm.size() * (comm.size() - 1) / 2);
        std::vector<int> expected_recv_counts(comm.size());
        std::iota(expected_recv_counts.begin(), expected_recv_counts.end(), 0);
        for (auto const& elem: result) {
            size_t source = elem.get_source();
            EXPECT_EQ(static_cast<double>(source) + 0.5, elem.get_payload());
            --expected_recv_counts[source];
        }
        EXPECT_THAT(expected_recv_counts, Each(0));
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}

TEST(AlltoallvGridTest, alltoallv_with_envelope_all_to_last_pe_source_destination_envelope) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()) + 0.5);
    std::vector<int>    send_counts(comm.size(), 0);
    send_counts[comm.size() - 1] = comm.rank_signed();

    constexpr auto envelope = MessageEnvelopeLevel::source_and_destination;
    auto result = grid_comm.alltoallv_with_envelope<envelope>(send_buf(input), kamping::send_counts(send_counts));

    if (comm.is_root(comm.size() - 1)) {
        EXPECT_EQ(result.size(), comm.size() * (comm.size() - 1) / 2);
        std::vector<int> expected_recv_counts(comm.size());
        std::iota(expected_recv_counts.begin(), expected_recv_counts.end(), 0);
        for (auto const& elem: result) {
            EXPECT_EQ(elem.get_destination(), comm.rank());
            size_t source = elem.get_source();
            EXPECT_EQ(static_cast<double>(source) + 0.5, elem.get_payload());
            --expected_recv_counts[source];
        }
        EXPECT_THAT(expected_recv_counts, Each(0));
    } else {
        EXPECT_EQ(result.size(), 0);
    }
}
