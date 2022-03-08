// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
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

#include <mpi.h>
#include <tuple>
#include <type_traits>

#include "kamping/checking_casts.hpp"
#include "kamping/kassert.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace kamping::internal {
template <typename Communicator>
class Alltoall {
public:
    template <typename... Args>
    auto alltoall(Args&&... args) {
        Communicator& comm = static_cast<Communicator&>(*this);
        // Get all parameters
        static_assert(
            internal::has_parameter_type<internal::ParameterType::send_buf, Args...>(),
            "Missing required parameter send_buf.");

        auto& send_buf_param       = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
        auto  send_buf             = send_buf_param.get();
        using send_value_type      = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        MPI_Datatype mpi_send_type = mpi_datatype<send_value_type>();

        using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<send_value_type>>{}));
        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(), args...);
        using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
        MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

        // Get the send and receive counts
        // TODO Use MPI_Type_size
        int send_count = throwing_cast<int>(send_buf.size / asserting_cast<size_t>(comm.size()));
        /// @todo test
        KTHROW(
            ((send_buf.size * sizeof(send_value_type)) % sizeof(recv_value_type)) == 0,
            "The specified receive type does not fit the supplied number of elements of the send type.");
        // Weird calculation because the user might use a different type for sending and receiving. For "normal" usage
        // this should be just send_buf.size
        size_t recv_buf_size = send_buf.size * sizeof(send_value_type) / sizeof(recv_value_type);
        int    recv_count    = throwing_cast<int>(recv_buf_size / asserting_cast<size_t>(comm.size()));

        int err = MPI_Alltoall(
            send_buf.ptr, send_count, mpi_send_type, recv_buf.get_ptr(recv_buf_size), recv_count, mpi_recv_type,
            comm.mpi_communicator());
        /// @todo throw correct exception with propagated error code
        KTHROW(err == MPI_SUCCESS);
        return MPIResult(
            std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
            internal::BufferCategoryNotUsed{});
    }

protected:
    Alltoall() {}
};
} // namespace kamping::internal
