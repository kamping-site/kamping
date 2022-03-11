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

#include <mpi.h>
#include <tuple>
#include <type_traits>

#include "kamping/checking_casts.hpp"
#include "kamping/kassert.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"

namespace kamping::internal {

/// @brief CRTP mixin class for \c MPI_Alltoall.
///
/// This class is only to be used as a super class of kamping::Communicator
template <typename Communicator>
class Alltoall {
public:
    /// @brief Wrapper for \c MPI_Alltoall
    ///
    /// This wrapper for \c MPI_Alltoall sends the same amount of data from each rank to each rank. The following
    /// buffers are required:
    /// - \ref kamping::send_buf() containing the data that is sent to each rank. This buffer has to be the same size at
    /// each rank and divisible by the size of the communicator. Each rank receives the same number of elements from
    /// this buffer. Rank 0 receives the first `<buffer size>/<communicator size>` elements, rank 1 the next, and so
    /// on. See
    /// TODO alltoallv if the amounts differ. The following buffers are optional:
    /// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, this buffer will contain
    /// the data received as specified for send_buf. The data received from rank 0 comes first, followed by the data
    /// received from rank 1, and so on.
    /// @tparam Args Automatically deducted template parameters.
    /// @param args All required and any number of the optional buffers described above.
    /// @return Result type wrapping the output buffer if not specified as input parameter.
    template <typename... Args>
    auto alltoall(Args&&... args) {
        static_assert(all_parameters_are_rvalues<Args...>);
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
        int send_count = throwing_cast<int>(send_buf.size / asserting_cast<size_t>(comm.size()));
        THROWING_KASSERT(mpi_send_type == mpi_recv_type, "The specified receive type does not match the send type.");

        size_t recv_buf_size = send_buf.size;
        int    recv_count    = throwing_cast<int>(recv_buf_size / asserting_cast<size_t>(comm.size()));
        KASSERT(send_count == recv_count, assert::light);

        // These KASSERTs are required to avoid a false warning from g++ in release mode
        auto* send_buf_ptr = send_buf.ptr;
        KASSERT(send_buf_ptr != nullptr);
        auto* recv_buf_ptr = recv_buf.get_ptr(recv_buf_size);
        KASSERT(recv_buf_ptr != nullptr);

        [[maybe_unused]] int err = MPI_Alltoall(
            send_buf.ptr, send_count, mpi_send_type, recv_buf.get_ptr(recv_buf_size), recv_count, mpi_recv_type,
            comm.mpi_communicator());

        THROW_IF_MPI_ERROR(err, MPI_Alltoall);
        return MPIResult(
            std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
            internal::BufferCategoryNotUsed{});
    }

protected:
    Alltoall() {}
};
} // namespace kamping::internal
