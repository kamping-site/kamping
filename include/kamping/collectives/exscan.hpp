
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

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_check.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

/// @brief Wrapper for \c MPI_Exscan.
///
/// This wraps \c MPI_Exscan, which is used to perform an exclusive prefix reduction on data distributed across the
/// calling processes. / \c exscan(...) returns in the recvbuf of the process with rank \c i, the reduction (calculated
/// according to the function op) of the values in the sendbufs of processes with ranks 0, ..., i (exclusive).
/// We set the value of the \c recv_buf on rank 0 to the value of on_rank_0 if provided. If \c on_rank_0 is not provided
/// and \c op is a build-in operation and we are working on a built in data-type, we set the value on rank 0 to the
/// identity of that operation. The type of operations supported, their semantics, and the constraints on send and
/// receive buffers are as for MPI_Reduce. The following parameters are required:
///  - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to be the same size at
///  each rank.
///  - \ref kamping::op() wrapping the operation to apply to the input.
///
/// The following parameter is required if the operation is not a build-in operation or the data-type is not a build-in
/// data type:
///  - \ref kamping::on_rank_0() containing the value that is returned in the \c recv_buf of rank 0.
///
///  The following parameters are optional:
///  - \ref kamping::recv_buf() containing a buffer for the output.
///  @tparam Args Automatically deducted template parameters.
///  @param args All required and any number of the optional buffers described above.
///  @return Result type wrapping the output buffer if not specified as input parameter.
template <typename... Args>
auto kamping::Communicator::exscan(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args, KAMPING_REQUIRED_PARAMETERS(send_buf, op), KAMPING_OPTIONAL_PARAMETERS(recv_buf, on_rank_0)
    );

    // Get the send buffer and deduce the send and recv value types.
    const auto& send_buf          = select_parameter_type<ParameterType::send_buf>(args...).get();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;
    KASSERT(
        is_same_on_all_ranks(send_buf.size()), "The send buffer has to be the same size on all ranks.",
        assert::light_communication
    );

    // Deduce the recv buffer type and get (if provided) the recv buffer or allocate one (if not provided).
    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<default_recv_value_type>>{}));
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
    recv_buf.resize(send_buf.size());
    recv_value_type* recv_buf_ptr = recv_buf.data();
    KASSERT(recv_buf_ptr != nullptr, assert::light);
    KASSERT(recv_buf.size() == send_buf.size(), assert::light);
    // send_buf.size() is equal on all ranks, as checked above.

    // Perform the MPI_Allreduce call and return.
    [[maybe_unused]] int err = MPI_Exscan(
        send_buf.data(),                      // sendbuf
        recv_buf_ptr,                         // recvbuf,
        asserting_cast<int>(send_buf.size()), // count
        mpi_datatype<send_value_type>(),      // datatype,
        operation.op(),                       // op
        mpi_communicator()                    // communicator
    );

    // MPI_Exscan leaves the recvbuf on rank 0 in an undefined state, we set it to the value of on_rank_0 if defined or
    // the identity of the operation otherwise (works only for build-in operations on build-in data types).
    if (rank() == 0) {
        constexpr bool on_root_param_provided = has_parameter_type<ParameterType::on_rank_0, Args...>();
        static_assert(
            on_root_param_provided || operation.is_builtin,
            "The user did not provide on_rank_0(...) and the operation is not built-in (at least on this type)."
        );
        if constexpr (on_root_param_provided) {
            const auto& on_rank_0_param = select_parameter_type<ParameterType::on_rank_0>(args...);
            KASSERT(
                (on_rank_0_param.size() == 1 || on_rank_0_param.size() == recv_buf.size()),
                "on_rank_0 has to either be of size 1 or of the same size as the recv_buf.", assert::light
            );
            // May be kamping::undefined
            if (on_rank_0_param.size() == 1) {
                std::fill_n(recv_buf.data(), recv_buf.size(), *on_rank_0_param.data());
            } else {
                std::copy_n(on_rank_0_param.data(), on_rank_0_param.size(), recv_buf.data());
            }
        } else if constexpr (operation.is_builtin) {
            std::fill_n(recv_buf.data(), recv_buf.size(), operation.identity());
        } else {
            assert(false);
        }
    }

    THROW_IF_MPI_ERROR(err, MPI_Reduce);
    return MPIResult(
        std::move(recv_buf), BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, BufferCategoryNotUsed{},
        BufferCategoryNotUsed{}
    );
}
