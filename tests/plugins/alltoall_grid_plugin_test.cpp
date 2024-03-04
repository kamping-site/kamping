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

#include "kamping/plugins/alltoall_grid_plugin.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::plugin;

using namespace grid_plugin_helpers;
template <typename T>
bool operator==(MessageEnvelope<T> const& lhs, MessageEnvelope<T> const& rhs) {
    return lhs.get_payload() == rhs.get_payload();
}

template <typename T>
bool operator==(MessageEnvelope<T, Source> const& lhs, MessageEnvelope<T, Source> const& rhs) {
    return std::make_tuple(lhs.get_payload(), lhs.get_source()) == std::make_tuple(rhs.get_payload(), rhs.get_source());
}

template <typename T>
bool operator==(
    MessageEnvelope<T, Source, Destination> const& lhs, MessageEnvelope<T, Source, Destination> const& rhs
) {
    return std::make_tuple(lhs.get_payload(), lhs.get_source(), lhs.get_destination())
           == std::make_tuple(rhs.get_payload(), rhs.get_source(), rhs.get_destination());
}

TEST(AlltoallvGridPluginTest, single_element_no_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto result = comm.alltoallv_grid(send_buf(input), kamping::send_counts(send_counts));
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(comm.rank_signed()));
}

TEST(AlltoallvGridPluginTest, single_element_source_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();

    std::vector<double> input(comm.size());
    std::iota(input.begin(), input.end(), 0.5);
    if (comm.is_root()) {
        for (auto const& elem: input) {
            std::cout << elem << std::endl;
        }
    }
    std::vector<int> send_counts(comm.size(), 1);

    constexpr auto envelope = MsgEnvelopeLevel::source;

    auto result = comm.alltoallv_grid<envelope>(send_buf(input), kamping::send_counts(send_counts));
    EXPECT_EQ(result.size(), comm.size());

    for (size_t i = 0; i < comm.size(); ++i) {
        auto it = std::find_if(result.begin(), result.end(), [&](auto const& msg) { return msg.get_source() == i; });
        EXPECT_NE(it, result.end());
        EXPECT_EQ(it->get_payload(), static_cast<double>(comm.rank()) + 0.5);
    }
}

TEST(AlltoallvGridPluginTest, single_element_source_destination_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();

    std::vector<double> input(comm.size());
    std::iota(input.begin(), input.end(), 0.5);
    if (comm.is_root()) {
        for (auto const& elem: input) {
            std::cout << elem << std::endl;
        }
    }
    std::vector<int> send_counts(comm.size(), 1);

    constexpr auto envelope = MsgEnvelopeLevel::source_and_destination;
    auto           result   = comm.alltoallv_grid<envelope>(send_buf(input), kamping::send_counts(send_counts));
    EXPECT_EQ(result.size(), comm.size());

    for (size_t i = 0; i < comm.size(); ++i) {
        auto it = std::find_if(result.begin(), result.end(), [&](auto const& msg) { return msg.get_source() == i; });
        EXPECT_NE(it, result.end());
        EXPECT_EQ(it->get_payload(), static_cast<double>(comm.rank()) + 0.5);
        EXPECT_EQ(it->get_destination(), comm.rank());
    }
}

TEST(AlltoallvGridPluginTest, last_to_all_pe_single_element_no_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();

    int                 last_pe = comm.size_signed() - 1;
    std::vector<double> input;
    std::vector<int>    send_counts(comm.size(), 0);
    if (comm.is_root(last_pe)) {
        int const count = comm.size_signed() * (comm.size_signed() - 1) / 2;
        input.resize(static_cast<size_t>(count), static_cast<double>(last_pe) + 0.5);
        std::iota(send_counts.begin(), send_counts.end(), 0);
    }

    auto result = comm.alltoallv_grid(send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(result.size(), comm.rank());
    EXPECT_THAT(result, Each(last_pe + 0.5));
}

TEST(AlltoallvGridPluginTest, last_to_all_pe_single_element_source_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();

    int                 last_pe = comm.size_signed() - 1;
    std::vector<double> input;
    std::vector<int>    send_counts(comm.size(), 0);
    if (comm.is_root(last_pe)) {
        int const count = comm.size_signed() * (comm.size_signed() - 1) / 2;
        input.resize(static_cast<size_t>(count), static_cast<double>(last_pe) + 0.5);
        std::iota(send_counts.begin(), send_counts.end(), 0);
    }

    constexpr auto envelope = MsgEnvelopeLevel::source;
    auto           result   = comm.alltoallv_grid<envelope>(send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(result.size(), comm.rank());
    for (auto const& elem: result) {
        EXPECT_EQ(elem.get_payload(), static_cast<double>(last_pe) + 0.5);
        EXPECT_EQ(elem.get_source(), last_pe);
    }
}

TEST(AlltoallvGridPluginTest, last_to_all_pe_single_element_source_destination_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();

    int                 last_pe = comm.size_signed() - 1;
    std::vector<double> input;
    std::vector<int>    send_counts(comm.size(), 0);
    if (comm.is_root(last_pe)) {
        int const count = comm.size_signed() * (comm.size_signed() - 1) / 2;
        input.resize(static_cast<size_t>(count), static_cast<double>(last_pe) + 0.5);
        std::iota(send_counts.begin(), send_counts.end(), 0);
    }

    constexpr auto envelope = MsgEnvelopeLevel::source_and_destination;
    auto           result   = comm.alltoallv_grid<envelope>(send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(result.size(), comm.rank());
    for (auto const& elem: result) {
        EXPECT_EQ(elem.get_payload(), static_cast<double>(last_pe) + 0.5);
        EXPECT_EQ(elem.get_source(), last_pe);
        EXPECT_EQ(elem.get_destination(), comm.rank());
    }
}

TEST(AlltoallvGridPluginTest, all_to_last_pe_single_element_no_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();
    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()) + 0.5);
    std::vector<int>    send_counts(comm.size(), 0);
    send_counts[comm.size() - 1] = comm.rank_signed();

    auto result = comm.alltoallv_grid(send_buf(input), kamping::send_counts(send_counts));

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

TEST(AlltoallvGridPluginTest, all_to_last_pe_single_element_source_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();
    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()) + 0.5);
    std::vector<int>    send_counts(comm.size(), 0);
    send_counts[comm.size() - 1] = comm.rank_signed();

    constexpr auto envelope = MsgEnvelopeLevel::source;
    auto           result   = comm.alltoallv_grid<envelope>(send_buf(input), kamping::send_counts(send_counts));

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

TEST(AlltoallvGridPluginTest, all_to_last_pe_single_element_source_destination_envelope) {
    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();
    std::vector<double> input(comm.rank(), static_cast<double>(comm.rank()) + 0.5);
    std::vector<int>    send_counts(comm.size(), 0);
    send_counts[comm.size() - 1] = comm.rank_signed();

    constexpr auto envelope = MsgEnvelopeLevel::source_and_destination;
    auto           result   = comm.alltoallv_grid<envelope>(send_buf(input), kamping::send_counts(send_counts));

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
