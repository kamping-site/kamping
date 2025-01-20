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

#include "kamping/mpi_datatype.hpp"
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

/// @brief Deduce the MPI_Datatype to use on the send and recv side.
/// If \ref kamping::send_type() is given, the \c MPI_Datatype wrapped inside will be used as send_type. Otherwise, the
/// \c MPI_datatype is derived automatically based on send_buf's underlying \c value_type.
///
/// If \ref kamping::recv_type()
/// is given, the \c MPI_Datatype wrapped inside will be used as recv_type. Otherwise, the \c MPI_datatype is derived
/// automatically based on recv_buf's underlying \c value_type.
///
/// @tparam send_value_type Value type of the send buffer.
/// @tparam recv_value_type Value type of the recv buffer.
/// @tparam recv_buf Type of the recv buffer.
/// @tparam Args Types of all arguments passed to the wrapped MPI call.
/// @param args All arguments passed to a wrapped MPI call.
/// @return Return a tuple containing the \c MPI send_type wrapped in a DataBuffer, the \c MPI recv_type wrapped in a
/// DataBuffer.
template <typename send_value_type, typename recv_value_type, typename recv_buf, typename... Args>
constexpr auto determine_mpi_datatypes(Args&... args) {
    // Some assertions:
    // If send/recv types are given, the corresponding count information has to be provided, too.
    constexpr bool is_send_type_given_as_in_param = is_parameter_given_as_in_buffer<ParameterType::send_type, Args...>;
    constexpr bool is_recv_type_given_as_in_param = is_parameter_given_as_in_buffer<ParameterType::recv_type, Args...>;
    if constexpr (is_send_type_given_as_in_param) {
        constexpr bool is_send_count_info_given =
            is_parameter_given_as_in_buffer<
                ParameterType::send_count,
                Args...> || is_parameter_given_as_in_buffer<ParameterType::send_counts, Args...> || is_parameter_given_as_in_buffer<ParameterType::send_recv_count, Args...>;
        static_assert(
            is_send_count_info_given,
            "If a custom send type is provided, send count(s) have to be provided, too."
        );
    }
    if constexpr (is_recv_type_given_as_in_param) {
        constexpr bool is_recv_count_info_given =
            is_parameter_given_as_in_buffer<
                ParameterType::recv_count,
                Args...> || is_parameter_given_as_in_buffer<ParameterType::recv_counts, Args...> || is_parameter_given_as_in_buffer<ParameterType::send_recv_count, Args...>;
        static_assert(
            is_recv_count_info_given,
            "If a custom recv type is provided, send count(s) have to be provided, too."
        );
    }
    // Recv buffer resize policy assertion
    constexpr bool do_not_resize_recv_buf = std::remove_reference_t<recv_buf>::resize_policy == no_resize;
    static_assert(
        !is_recv_type_given_as_in_param || do_not_resize_recv_buf,
        "If a custom recv type is given, kamping is not able to deduce the correct size of the recv buffer. "
        "Therefore, a sufficiently large recv buffer (with resize policy \"no_resize\") must be provided by the user."
    );

    // Get the send/recv types
    using default_mpi_send_type = decltype(kamping::send_type_out());
    using default_mpi_recv_type = decltype(kamping::recv_type_out());

    auto mpi_send_type =
        internal::select_parameter_type_or_default<internal::ParameterType::send_type, default_mpi_send_type>(
            std::make_tuple(),
            args...
        )
            .construct_buffer_or_rebind();
    if constexpr (!is_send_type_given_as_in_param) {
        if constexpr (std::is_same_v<send_value_type, unused_tparam>) {
            mpi_send_type.underlying() = MPI_DATATYPE_NULL;
        } else {
            mpi_send_type.underlying() = mpi_datatype<send_value_type>();
        }
    }

    auto mpi_recv_type =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_type, default_mpi_recv_type>(
            std::make_tuple(),
            args...
        )
            .construct_buffer_or_rebind();
    if constexpr (!is_recv_type_given_as_in_param) {
        mpi_recv_type.underlying() = mpi_datatype<recv_value_type>();
    }

    return std::tuple{std::move(mpi_send_type), std::move(mpi_recv_type)};
}

/// @brief Deduce the MPI_Datatype to use as send_recv_type in a collective operation which accepts only one parameter
/// of MPI_Datatype instead of (possibly) distinct send and recv types. If \ref kamping::send_recv_type() is given, the
/// \c MPI_Datatype wrapped inside will be used as send_recv_type. Otherwise, the \c MPI_datatype is derived
/// automatically based on send_buf's underlying \c value_type.
///
/// @tparam send_or_send_recv_value_type Value type of the send(_recv) buffer.
/// @tparam recv_buf Value type of the send buffer.
/// @tparam recv_or_send_recv_buf Type of the (send_)recv buffer.
/// @tparam Args Types of all arguments passed to the wrapped MPI call.
/// @param args All arguments passed to a wrapped MPI call.
/// @return Return the \c MPI send_type wrapped in a DataBuffer. This is either an lvalue reference to the
/// send_recv_type DataBuffer if the send_recv_type is provided by the user or a newly created send_recv_type DataBuffer
/// otherwise.
template <typename send_or_send_recv_value_type, typename recv_or_send_recv_buf, typename... Args>
constexpr auto determine_mpi_send_recv_datatype(Args&... args)
    -> decltype(internal::select_parameter_type_or_default<
                    internal::ParameterType::send_recv_type,
                    decltype(kamping::send_recv_type_out())>(std::make_tuple(), args...)
                    .construct_buffer_or_rebind()) {
    // Some assertions:
    // If a send_recv type is given, the corresponding count information has to be provided, too.
    constexpr bool is_send_recv_type_given_as_in_param =
        is_parameter_given_as_in_buffer<ParameterType::send_recv_type, Args...>;
    if constexpr (is_send_recv_type_given_as_in_param) {
        constexpr bool is_send_recv_count_given =
            is_parameter_given_as_in_buffer<ParameterType::send_recv_count, Args...>;
        static_assert(
            is_send_recv_count_given,
            "If a custom send_recv type is provided, the send_recv count has to be provided, too."
        );
    }
    // Recv buffer resize policy assertion
    constexpr bool do_not_resize_recv_buf = std::remove_reference_t<recv_or_send_recv_buf>::resize_policy == no_resize;
    static_assert(
        !is_send_recv_type_given_as_in_param || do_not_resize_recv_buf,
        "If a custom send_recv type is given, kamping is not able to deduce the correct size of the "
        "recv/send_recv buffer. "
        "Therefore, a sufficiently large recv/send_recv buffer (with resize policy \"no_resize\") must be provided by "
        "the user."
    );

    // Get the send_recv type
    using default_mpi_send_recv_type = decltype(kamping::send_recv_type_out());

    auto mpi_send_recv_type =
        internal::select_parameter_type_or_default<internal::ParameterType::send_recv_type, default_mpi_send_recv_type>(
            std::make_tuple(),
            args...
        )
            .construct_buffer_or_rebind();

    // assure that our expectation about the return value value category (lvalue or pr-value) is true. This ensures
    // that the return value of the function does not become a dangling rvalue reference bound to a function-local
    // object.
    static_assert(
        !std::is_rvalue_reference_v<decltype(mpi_send_recv_type)>,
        "mpi_send_type is either a lvalue reference (in this case it returned by reference), or a non-reference type "
        "(in "
        "this case it is returned by value)."
    );

    if constexpr (!is_send_recv_type_given_as_in_param) {
        mpi_send_recv_type.underlying() = mpi_datatype<send_or_send_recv_value_type>();
    }

    return mpi_send_recv_type;
}

} // namespace kamping::internal
