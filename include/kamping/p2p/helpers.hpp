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
/// @brief Deduce the MPI_Datatype to use as send_type in a p2p send operation.If \ref kamping::send_type() is given,
/// the \c MPI_Datatype wrapped inside will be used as send_type. Otherwise, the \c MPI_datatype is derived
/// automatically based on send_buf's underlying \c value_type.
///
/// @tparam send_value_type Value type of the send buffer.
/// @tparam Args Types of all arguments passed to the wrapped MPI call.
/// @param args All arguments passed to a wrapped MPI call.
/// @return Return the \c MPI send_type wrapped in a DataBuffer. This is either an lvalue reference to the
/// send_type DataBuffer if the send_type is provided by the user or a newly created send_type DataBuffer
/// otherwise.
template <typename send_value_type, typename... Args>
constexpr auto determine_mpi_send_datatype(Args&... args)
    -> decltype(internal::select_parameter_type_or_default<
                    internal::ParameterType::send_type,
                    decltype(kamping::send_type_out())>(std::make_tuple(), args...)
                    .construct_buffer_or_rebind()) {
    // Some assertions:
    // If a send_type is given, the send_count information has to be provided, too.
    constexpr bool is_send_type_given_as_in_param = is_parameter_given_as_in_buffer<ParameterType::send_type, Args...>;
    if constexpr (is_send_type_given_as_in_param) {
        constexpr bool is_send_count_given = is_parameter_given_as_in_buffer<ParameterType::send_count, Args...>;
        static_assert(
            is_send_count_given,
            "If a custom send type is provided, the send count has to be provided, too."
        );
    }
    // Get the send type
    using default_mpi_send_type = decltype(kamping::send_type_out());

    auto mpi_send_type =
        internal::select_parameter_type_or_default<internal::ParameterType::send_type, default_mpi_send_type>(
            std::make_tuple(),
            args...
        )
            .construct_buffer_or_rebind();

    // assure that our expectation about the return value value category (lvalue or pr-value) is true. This ensures
    // that the return value of the function does not become a dangling rvalue reference bound to a function-local
    // object.
    static_assert(
        !std::is_rvalue_reference_v<decltype(mpi_send_type)>,
        "mpi_send_type is either a lvalue reference (in this case it returned by reference), or a non-reference type "
        "(in "
        "this case it is returned by value)."
    );

    if constexpr (!is_send_type_given_as_in_param) {
        mpi_send_type.underlying() = mpi_datatype<send_value_type>();
    }

    return mpi_send_type;
}

/// @brief Deduce the MPI_Datatype to use as recv_type in a p2p recv operation.If \ref kamping::recv_type() is given,
/// the \c MPI_Datatype wrapped inside will be used as recv_type. Otherwise, the \c MPI_datatype is derived
/// automatically based on recv_buf's underlying \c value_type.
///
/// @tparam recv_value_type Value type of the recv buffer.
/// @tparam recv_buf Type of the recv buffer.
/// @tparam Args Types of all arguments passed to the wrapped MPI call.
/// @param args All arguments passed to a wrapped MPI call.
/// @return Return the \c MPI recv_type wrapped in a DataBuffer. This is either an lvalue reference to the
/// recv_type DataBuffer if the recv_type is provided by the user or a newly created recv_type DataBuffer
/// otherwise.
template <typename recv_value_type, typename recv_buf, typename... Args>
constexpr auto determine_mpi_recv_datatype(Args&... args)
    -> decltype(internal::select_parameter_type_or_default<
                    internal::ParameterType::recv_type,
                    decltype(kamping::recv_type_out())>(std::make_tuple(), args...)
                    .construct_buffer_or_rebind()) {
    // Get the recv type
    using default_mpi_recv_type = decltype(kamping::recv_type_out());

    auto mpi_recv_type =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_type, default_mpi_recv_type>(
            std::make_tuple(),
            args...
        )
            .construct_buffer_or_rebind();

    // assure that our expectation about the return value value category (lvalue or pr-value) is true. This ensures
    // that the return value of the function does not become a dangling rvalue reference bound to a function-local
    // object.
    static_assert(
        !std::is_rvalue_reference_v<decltype(mpi_recv_type)>,
        "mpi_recv_type is either a lvalue reference (in this case it returned by reference), or a non-reference type "
        "(in "
        "this case it is returned by value)."
    );

    constexpr bool is_recv_type_given_as_in_param = is_parameter_given_as_in_buffer<ParameterType::recv_type, Args...>;
    // Recv buffer resize policy assertion
    constexpr bool do_not_resize_recv_buf = std::remove_reference_t<recv_buf>::resize_policy == no_resize;
    static_assert(
        !is_recv_type_given_as_in_param || do_not_resize_recv_buf,
        "If a custom recv type is given, kamping is not able to deduce the correct size of the "
        "recv buffer. "
        "Therefore, a sufficiently large recv buffer (with resize policy \"no_resize\") must be provided by "
        "the user."
    );

    if constexpr (!is_recv_type_given_as_in_param) {
        mpi_recv_type.underlying() = mpi_datatype<recv_value_type>();
    }

    return mpi_recv_type;
}
} // namespace kamping::internal
