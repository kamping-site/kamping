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

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"
#include <mpi.h>

namespace kamping {

/// @brief Wrapper for \c MPI_Gather
///
/// This wrapper for \c MPI_Gather sends the same amount of data from each rank to a root. The following buffers are
/// required:
/// - \ref kamping::send_buf() containing the data that is sent to the root. This buffer has to be the same size at each
/// rank. See TODO gather_v if the amounts differ. The following buffers are optional:
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c Communicator is
/// used, see root().
/// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, at the root, this buffer will contain all
/// data from all send buffers. At all other ranks, the buffer will have size 0.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <typename... Args>
auto Communicator::gather(Args&&... args) {
    auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf        = send_buf_param.get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;

    using default_recv_buf_type = decltype(kamping::recv_buf(NewContainer<std::vector<send_value_type>>{}));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(), args...);
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::Root>(
        std::tuple(_root), args...);

    auto mpi_send_type = mpi_datatype<send_value_type>();
    auto mpi_recv_type = mpi_datatype<recv_value_type>();

    size_t recv_size     = (_rank == root.rank()) ? send_buf.size : 0;
    size_t recv_buf_size = asserting_cast<size_t>(_size) * recv_size;

    // Check if the root is valid, before we try any communication
    KTHROW(is_valid_rank(root.rank()), "Invalid rank as root.");
    KTHROW(mpi_send_type == mpi_recv_type, "The specified receive type does not match the send type.");

    [[maybe_unused]] auto check_equal_sizes = [&]() {
        std::vector<size_t> result(asserting_cast<size_t>(_size), 0);
        size_t              local_size = asserting_cast<size_t>(send_buf.size);
        MPI_Gather(
            &local_size, 1, mpi_datatype<size_t>(), result.data(), 1, mpi_datatype<size_t>(), root.rank(), _comm);
        for (size_t i = 1; i < result.size(); ++i) {
            if (result[i] != result[i - 1]) {
                return false;
            }
        }
        return true;
    };
    KASSERT(
        check_equal_sizes(),
        "All PEs have to send the same number of elements. Use gatherv, if you want to send a different number of "
        "elements.",
        4);

    // error code can be unused if KTHROW is removed at compile time
    [[maybe_unused]] int err = MPI_Gather(
        send_buf.ptr, asserting_cast<int>(send_buf.size), mpi_send_type, recv_buf.get_ptr(recv_buf_size),
        asserting_cast<int>(recv_size), mpi_recv_type, root.rank(), _comm);
    KTHROW(err == MPI_SUCCESS);
    return MPIResult(
        std::move(recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
        internal::BufferCategoryNotUsed{});
}

} // namespace kamping
