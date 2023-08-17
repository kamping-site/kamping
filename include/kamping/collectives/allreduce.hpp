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

#include <tuple>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @brief Wrapper for \c MPI_Allreduce; which is semantically a reduction followed by a broadcast.
///
/// This wraps \c MPI_Allreduce. The operation combines the elements in the input buffer provided via \c
/// kamping::send_buf() and returns the combined value on all ranks. The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to be the same size at
/// each rank.
/// - \ref kamping::op() wrapping the operation to apply to the input.
///
/// The following parameters are optional:
/// - \ref kamping::recv_buf() containing a buffer for the output.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::allreduce(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, op), KAMPING_OPTIONAL_PARAMETERS(recv_buf));

    // Get the send buffer and deduce the send and recv value types.
    auto const& send_buf          = select_parameter_type<ParameterType::send_buf>(args...).get();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;
    KASSERT(
        is_same_on_all_ranks(send_buf.size()),
        "The send buffer has to be the same size on all ranks.",
        assert::light_communication
    );

    // Deduce the recv buffer type and get (if provided) the recv buffer or allocate one (if not provided).
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto&& recv_buf =
        select_parameter_type_or_default<ParameterType::recv_buf, default_recv_buf_type>(std::tuple(), args...);
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    static_assert(
        std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
        "Types of send and receive buffers do not match."
    );

    // Get the operation used for the reduction. The signature of the provided function is checked while building.
    auto& operation_param = select_parameter_type<ParameterType::op>(args...);
    auto  operation       = operation_param.template build_operation<send_value_type>();

    // Resize the recv buffer to the same size as the send buffer; get the pointer needed for the MPI call.
    send_value_type* recv_buf_ptr = nullptr;
    recv_buf.resize(send_buf.size());
    recv_buf_ptr = recv_buf.data();
    KASSERT(recv_buf_ptr != nullptr, assert::light);
    KASSERT(recv_buf.size() == send_buf.size(), assert::light);
    // send_buf.size() is equal on all ranks, as checked above.

    // Perform the MPI_Allreduce call and return.
    [[maybe_unused]] int err = MPI_Allreduce(
        send_buf.data(),                      // sendbuf
        recv_buf_ptr,                         // recvbuf,
        asserting_cast<int>(send_buf.size()), // count
        mpi_datatype<send_value_type>(),      // datatype,
        operation.op(),                       // op
        mpi_communicator()                    // communicator
    );

    THROW_IF_MPI_ERROR(err, MPI_Reduce);
    return make_mpi_result(std::move(recv_buf));
}

/// @brief Wrapper for \c MPI_Allreduce; which is semantically a reduction followed by a broadcast.
///
/// This wrapper for \c MPI_Allreduce sends a single value from the root to all other ranks. Calling allreduce_single()
/// is a shorthand for calling allreduce() with a \ref send_buf of size 1. It always issues only a single
/// <code>MPI_Allreduce</code> call, as no receive counts have to be exchanged.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to be of size 1 on each
/// rank.
/// - \ref kamping::op() wrapping the operation to apply to the input.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return The single output value.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::allreduce_single(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, op), KAMPING_OPTIONAL_PARAMETERS());

    KASSERT(
        select_parameter_type<ParameterType::send_buf>(args...).get().size() == 1ul,
        "The send buffer has to be of size 1 on all ranks.",
        assert::light
    );
    using value_type =
        typename std::remove_reference_t<decltype(select_parameter_type<ParameterType::send_buf>(args...))>::value_type;

    return this->allreduce(std::forward<Args>(args)..., recv_buf(alloc_new<value_type>)).extract_recv_buffer();
}
