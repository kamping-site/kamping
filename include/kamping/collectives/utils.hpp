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
//

#pragma once
#include "kamping/has_member.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"

namespace kamping::internal {

/// @brief Deduce the MPI_Datatype to use on the send and recv side.
/// If \ref kamping::send_type() is given, the \c MPI_Dataype wrapped inside will be used as send_type. Otherwise, the
/// \c MPI_datatype is derived automatically based on send_buf's underlying \c value_type.
///
/// If \ref kamping::recv_type()
/// is given, the \c MPI_Dataype wrapped inside will be used as recv_type. Otherwise, the \c MPI_datatype is derived
/// automatically based on recv_buf's underlying \c value_type.
///
/// @tparam send_value_type Value type of the send buffer.
/// @tparam recv_value_type Value type of the recv buffer.
/// @param args All arguments passed to a wrapped MPI call.
/// @return Return a tuple containing the \c MPI send_type wrapped in a DataBuffer, the \c MPI recv_type wrapped in a
/// DataBuffer.
template <typename send_value_type, typename recv_value_type, typename recv_buf, typename... Args>
auto determine_mpi_datatypes(Args&... args) {
    using default_mpi_send_type = decltype(kamping::send_type_out());
    using default_mpi_recv_type = decltype(kamping::recv_type_out());

    auto&& mpi_send_type =
        internal::select_parameter_type_or_default<internal::ParameterType::send_type, default_mpi_send_type>(
            std::make_tuple(),
            args...
        );

    // cannot do this via default construction in previous call to select_parameter_type_or_default due to
    // send_type_out() as possible input parameter.
    if constexpr (has_to_be_computed<decltype(mpi_send_type)>) {
        *mpi_send_type.data() = mpi_datatype<send_value_type>();
    }

    auto&& mpi_recv_type =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_type, default_mpi_recv_type>(
            std::make_tuple(),
            args...
        );

    // cannot do this via default construction in previous call to select_parameter_type_or_default due to
    // recv_type_out() as possible input parameter.
    if constexpr (has_to_be_computed<decltype(mpi_recv_type)>) {
        *mpi_recv_type.data() = mpi_datatype<recv_value_type>();
    }

    static_assert(
        internal::has_to_be_computed<decltype(mpi_recv_type)> || !internal::has_to_be_allocated_by_library<recv_buf>,
        "If the recv_type() parameter is given, kamping does not resize the recv buffer. Therefore, the recv "
        "buffer must be given as a parameter with a preallocated size sufficient for the received elements."
    );

    return std::make_tuple(std::move(mpi_send_type), std::move(mpi_recv_type));
}

/// @brief Compute the required recv buffer size with minimal overhead, which is the max_i(recv_counts[i] +
/// recv_displs[i]) for 0 <= i < communicator size.
///
/// @tparam recv_displs_are_given Specifies whether recv displs are explicitly passed to the wrapped \c MPI call.
/// @tparam recv_buf_is_resizable Specifies whether recv buffer is resizable (and therefore shall be resized if
/// necessary).
/// @tparam RecvCounts Type of the wrapped recv_counts.
/// @tparam RecvDispls Type of the wrapped recv_displs.
/// @param recv_counts Given or computed recv counts.
/// @param recv_displs Given or computed recv displs.
/// @return Necessary recv buf size to which the recv_buffer shall be resized.
template <bool recv_displs_are_given, bool recv_buf_is_resizable, typename RecvCounts, typename RecvDispls>
int compute_necessary_recv_buf_size(RecvCounts const& recv_counts, RecvDispls const& recv_displs) {
    KASSERT(recv_counts.size() == recv_displs.size(), "The number of recv counts and recv displacements differs.");
    if constexpr (!recv_displs_are_given) {
        return *(recv_counts.data() + recv_counts.size() - 1) + // Last element of recv_counts
               *(recv_displs.data() + recv_displs.size() - 1);  // Last element of recv_displs
    }
    int size = 0;
    if constexpr (recv_buf_is_resizable) {
        for (size_t i = 0; i < recv_displs.size(); ++i) {
            int recv_count = *(recv_counts.data() + i);
            int recv_displ = *(recv_displs.data() + i);
            size           = std::max(size, (recv_count + recv_displ));
        }
    }
    return size;
}
} // namespace kamping::internal
