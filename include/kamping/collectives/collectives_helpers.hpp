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

#pragma once

#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameters.hpp"

namespace kamping::internal {
/// @brief Compute the required size of the recv buffer in vectorized communication (i.e. \c MPI operation that take a
/// receive displacements). If recv displs are provided by the user the required size is the sum of the last entries of
/// the recv_counts and the recv_displs buffers. Otherwise we have to compute the elementwise maximum of both buffers to
/// obtain a minimum required recv buf size.
///
/// @tparam RecvCounts Type of the recv counts buffer.
/// @tparam RecvDispls Type of the recv displs buffer.
/// @param recv_counts Recv counts buffer.
/// @param recv_displs Recv displs buffer.
/// @param comm_size Size of the communicator.
/// @return Required size of the recv buffer.
template <typename RecvCounts, typename RecvDispls>
size_t compute_required_recv_buf_size_in_vectorized_communication(
    RecvCounts const& recv_counts, RecvDispls const& recv_displs, size_t comm_size
) {
    constexpr bool do_calculate_recv_displs = internal::has_to_be_computed<RecvDispls>;
    if constexpr (do_calculate_recv_displs) {
        // If recv displs are not provided as a parameter, they are monotonically increasing. In this case, it is
        // safe to deduce the required recv_buf size by only considering  the last entry of recv_counts and
        // recv_displs.
        int recv_buf_size = *(recv_counts.data() + comm_size - 1) + // Last element of recv_counts
                            *(recv_displs.data() + comm_size - 1);  // Last element of recv_displs
        return asserting_cast<size_t>(recv_buf_size);
    } else {
        // If recv displs are user provided, they do not need to be monotonically increasing. Therefore, we have to
        // compute the maximum of recv_displs and recv_counts from each rank to provide a receive buffer large
        // enough to be able to receive all elements. This O(p) computation is only executed if the user wants
        // kamping to resize the receive buffer.
        int recv_buf_size = 0;
        for (size_t i = 0; i < comm_size; ++i) {
            recv_buf_size = std::max(recv_buf_size, *(recv_counts.data() + i) + *(recv_displs.data() + i));
        }
        return asserting_cast<size_t>(recv_buf_size);
    }
}
} // namespace kamping::internal
