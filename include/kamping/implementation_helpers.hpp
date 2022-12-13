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

#pragma once

#include "kamping/parameter_objects.hpp"

namespace kamping::internal {
/// @brief Checks whether a RankDataBuffer contains a valid rank in the given communicator.
///
/// Can also be configured to accept RankType::null or RankType::any.
///
/// @param rank_data_buffer The RankDataBuffer encapsulating the rank to check.
/// @param comm The Communicator to check for validity in.
/// @param allow_null Whether this function should return true for RankType::null.
/// @param allow_any Whether this function should return true for RankType::any.
/// @tparam RankDataBufferClass The template instantiation of RankDataBuffer.
/// @tparam Comm The template instantiation of Communicator.
template <typename RankDataBufferClass, typename Comm>
constexpr bool is_valid_rank_in_comm(
    RankDataBufferClass const& rank_data_buffer,
    Comm const&                comm,
    bool const                 allow_null = false,
    bool const                 allow_any  = false
) {
    constexpr auto rank_type = std::remove_reference_t<decltype(rank_data_buffer)>::rank_type;
    if constexpr (rank_type == RankType::value) {
        return comm.is_valid_rank(rank_data_buffer.rank_signed());
    } else if constexpr (rank_type == RankType::null) {
        return allow_null;
    } else if constexpr (rank_type == RankType::any) {
        return allow_any;
    }
    return false;
}

} // namespace kamping::internal
