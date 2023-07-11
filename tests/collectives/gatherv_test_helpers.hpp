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

/// @file
/// @brief Object generating the expected output that is used in multiple (all)gatherv tests.

#pragma once
#include <numeric>
#include <vector>

#include "kamping/communicator.hpp"

namespace testing {
/// @brief Generates the expected receive buffer, receive counts and receive displacements buffers for receiving ranks
/// when each rank sends rank times its rank in a (all) gatherv operation.
struct ExpectedBuffersForRankTimesRankGathering {
    /// @brief Generates expected receive buffer on receiving ranks.
    ///
    /// @tparam T Datatype to which the ranks will be converted.
    /// @tparam Container Type of the buffer.
    /// @param comm Communicator which will be used in the scenario.
    /// @return Receive buffer.
    template <typename T, template <typename...> typename Container = std::vector>
    static auto recv_buffer_on_receiving_ranks(kamping::Communicator<> const& comm) {
        Container<T> container;
        for (size_t i = 0; i < comm.size(); ++i) {
            std::fill_n(std::back_inserter(container), i, static_cast<T>(i));
        }
        return container;
    }

    /// @brief Generates expected receive counts on receiving ranks.
    ///
    /// @tparam Container Type of the buffer.
    /// @param comm Communicator which will be used in the scenario.
    /// @return Receive counts.
    template <template <typename...> typename Container = std::vector>
    static auto recv_counts_on_receiving_ranks(kamping::Communicator<> const& comm) {
        Container<int> recv_counts(comm.size());
        std::iota(recv_counts.begin(), recv_counts.end(), 0);
        return recv_counts;
    }

    /// @brief Generates expected receive displacements on receiving ranks.
    ///
    /// @tparam Container Type of the buffer.
    /// @param comm Communicator which will be used in the scenario.
    /// @return Receive displacements.
    template <template <typename...> typename Container = std::vector>
    static auto recv_displs_on_receiving_ranks(kamping::Communicator<> const& comm) {
        auto           recv_counts = recv_counts_on_receiving_ranks(comm);
        Container<int> recv_displs(comm.size());
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);
        return recv_displs;
    }
};
} // namespace testing
