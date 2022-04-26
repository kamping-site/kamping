// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <tuple>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_check.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

/// @brief Wrapper for \c MPI_Reduce.
///
/// This wraps \c MPI_Reduce. The operation combines the elements in the input buffer provided via \c
/// kamping::send_buf() and returns the combined value on the root rank. The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to be the same size at
/// each rank.
/// - \ref kamping::op() wrapping the operation to apply to the input.
///
/// The following parameters are optional:
/// - \ref kamping::recv_buf() containing a buffer for the output.
/// - \ref kamping::root() the root rank. If not set, the default root process of the communicator will be used.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <typename... Args>
auto kamping::Communicator::reduce(Args&&... args) {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args, KAMPING_REQUIRED_PARAMETERS(send_buf, op), KAMPING_OPTIONAL_PARAMETERS(recv_buf, root));

    // Get all parameters
    auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::Root>(
        std::tuple(this->root()), args...);

    const auto& send_buf          = internal::select_parameter_type<internal::ParameterType::send_buf>(args...).get();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;

    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<default_recv_value_type>>{}));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(), args...);
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    auto& operation_param = internal::select_parameter_type<internal::ParameterType::op>(args...);
    auto  operation       = operation_param.template build_operation<send_value_type>();

    // Check parameters
    static_assert(
        std::is_same_v<std::remove_const_t<send_value_type>, recv_value_type>,
        "Types of send and receive buffers do not match.");
    MPI_Datatype type = mpi_datatype<send_value_type>();

    KASSERT(is_valid_rank(root.rank()), "The provided root rank is invalid.");

    send_value_type* recv_buf_ptr = nullptr;
    if (rank() == root.rank()) {
        recv_buf.resize(send_buf.size());
        recv_buf_ptr = recv_buf.data();
    }
    [[maybe_unused]] int err = MPI_Reduce(
        send_buf.data(), recv_buf_ptr, asserting_cast<int>(send_buf.size()), type, operation.op(), root.rank_signed(),
        mpi_communicator());

    THROW_IF_MPI_ERROR(err, MPI_Reduce);
    return MPIResult(
        std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
        internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{});
}
