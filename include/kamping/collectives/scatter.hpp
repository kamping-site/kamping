// This file is part of KaMPI.ng
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

#include <mpi.h>
#include <type_traits>

#include "kamping/checking_casts.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/kassert.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"

namespace kamping::internal {
template <typename Communicator>
class Scatter {
public:
    template <typename... Args>
    auto scatter(Args&&... args) {
        // Required parameter: send_buf()
        static_assert(
            internal::has_parameter_type<internal::ParameterType::send_buf, Args...>(),
            "Missing required parameter send_buf.");

        auto send_buf              = internal::select_parameter_type<internal::ParameterType::send_buf>(args...).get();
        using send_value_type      = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        MPI_Datatype mpi_send_type = mpi_datatype<send_value_type>();

        // Compute sendcount based on the size of the sendbuf
        KASSERT(
            send_buf.size % this->comm().size() == 0,
            "Size of the send buffer (" << send_buf.size << ") is not divisible by the number of PEs (" << comm().size()
                                        << ") in the communicator.");
        int const send_count = asserting_cast<int>(send_buf.size / comm().size());

        // Optional parameter: root()
        // Default: communicator root
        int const root = internal::select_parameter_type_or_default<internal::ParameterType::root>(
                             std::tuple(comm().root()), args...)
                             .root();

        // Optional parameter: recv_buf()
        // Default: allocate new container
        using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<send_value_type>>{}));
        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(), args...);
        using recv_value_type      = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
        MPI_Datatype mpi_recv_type = mpi_datatype<recv_value_type>();

        // Optional parameter: recv_count()
        // Default: compute value based on send_buf.size on root
        int recv_count = 0;

        if (internal::has_parameter_type<internal::ParameterType::recv_count>(args...)) {
            recv_count = internal::select_parameter_type<internal::ParameterType::recv_count>(args...).recv_count();

            // Validate against send_count
            KASSERT(
                recv_count == bcast_recv_count(send_count, root), "Specified recv count does not match the send count.",
                assert::light_communication);
        } else {
            // Broadcast send_count to get recv_count
            recv_count = this->bcast_recv_count(send_count, root);
        }

        auto* send_buf_ptr = send_buf.ptr;
        auto* recv_buf_ptr = recv_buf.get_ptr(recv_count);

        [[maybe_unused]] int const err = MPI_Scatter(
            send_buf_ptr, send_count, mpi_send_type, recv_buf_ptr, recv_count, mpi_recv_type, root,
            comm().mpi_communicator());
        THROW_IF_MPI_ERROR(err, MPI_Scatter);

        return MPIResult(
            std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
            internal::BufferCategoryNotUsed{});
    }

private:
    int bcast_recv_count(int const bcast_value, int root) {
        int                        bcast_result = bcast_value;
        [[maybe_unused]] int const result       = MPI_Bcast(&bcast_result, 1, MPI_INT, root, comm().mpi_communicator());
        THROW_IF_MPI_ERROR(result, MPI_Bcast);
        return bcast_result;
    }

    Communicator const& comm() const {
        return static_cast<Communicator const&>(*this);
    }
};
} // namespace kamping::internal
