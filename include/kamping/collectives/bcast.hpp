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

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_check.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/parameter_type_definitions.hpp"

/// @brief Wrapper for \c MPI_Bcast
///
/// This wrapper for \c MPI_Bcast sends data from the root to all other ranks.
/// The following buffers are required:
/// - \ref kamping::send_buf() containing the data that is sent to the other ranks.
/// The following parameters are optional:
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c
/// Communicator is used, see root().
/// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, at all other ranks, this buffer
/// will contain the data from the root send buffer.
/// @todo Describe what happens at the root
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <typename... Args>
auto kamping::Communicator::bcast(Args... args) {
    using namespace ::kamping::internal;

    static_assert(
        internal::all_parameters_are_rvalues<Args...>,
        "All parameters have to be passed in as rvalue references, meaning that you must not hold a variable "
        "returned by the named parameter helper functions like recv_buf().");

    // Check and get all parameters.

    // The parameter send_recv_buf() is required on all processes.
    static_assert(
        has_parameter_type<internal::ParameterType::send_recv_buf, Args...>(),
        "Missing required parameter send_recv_buf.");

    const auto& send_recv_buf = internal::select_parameter_type<internal::ParameterType::send_recv_buf>(args...).get();
    using value_type          = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;

    auto&& root = internal::select_parameter_type_or_default<internal::ParameterType::root, internal::Root>(
        std::tuple(this->root()), args...);

    auto mpi_value_type = mpi_datatype<value_type>();

    // Conduct some validity check on the parmeters.
    KASSERT(this->is_valid_rank(root.rank()), "Invalid rank as root.", assert::light);

    if (this->is_root(root.rank())) {
        KASSERT(send_recv_buf.size() > 0ul, "The send_recv_buf() on the root process is empty.", assert::light);

        KASSERT(
            !std::is_const_v<decltype(send_recv_buf)>,
            "This rank has to be either root or have a non-const send_recv_buf.", assert::light);
    }

    /// @todo Implement and test that passing a const-buffer is allowed on the root process but not on all other
    /// processes.

    /// @todo Once we decided on how to handle different buffer sizes passed to different processes, implement this
    /// here.
    // KASSERT(
    //     recv_buf_large_enough_on_all_processes(send_recv_buf, root.rank()),
    //     "The receive buffer is too small on at least one rank.", assert::light_communication);

    // Perform the broadcast. The error code is unused if KTHROW is removed at compile time.
    [[maybe_unused]] int err = MPI_Bcast(
        send_recv_buf.data(),                      // buffer*
        asserting_cast<int>(send_recv_buf.size()), // count
        mpi_value_type,                            // datatype
        root.rank_signed(),                        // root
        this->mpi_communicator()                   // MPI_Comm comm
    );
    THROW_IF_MPI_ERROR(err, MPI_Bcast);

    return MPIResult(
        std::move(send_recv_buf), internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{},
        internal::BufferCategoryNotUsed{}, internal::BufferCategoryNotUsed{});
} // namespace kamping::internal

// /// @brief Checks if the receive buffer is large enough to receive all elements on all ranks.
// ///
// /// Broadcasts the size of the send buffer (which is equal to the recv_buf) from the root rank,
// /// performs local comparison and collects the result using an allreduce.
// /// @param send_recv_buf The send buffer on root, the receive buffer on all other ranks.
// /// @param root The rank of the root process.
// /// @todo Once we decided on which ranks to notify of failed exceptions and the CRTP helper is there,
// /// implement this.
// template <typename RecvBuf>
// bool recv_buf_large_enough_on_all_processes(RecvBuf const& send_recv_buf, int const root) const {
//     uint64_t size = send_recv_buf.size();
//     MPI_Bcast(
//         &size,                                // src/dest buffer
//         1,                                    // size
//         mpi_datatype<decltype(size)>(),       // datatype
//         root,                                 // root
//         this->underlying().mpi_communicator() // communicator
//     );
//     bool const local_buffer_large_enough = size <= send_recv_buf.size();
//     bool       every_buffer_large_enough;
//     MPI_Allreduce(
//         &local_buffer_large_enough,           // src buffer
//         &every_buffer_large_enough,           // dest buffer
//         1,                                    // count
//         mpi_datatype<bool>(),                 // datatype
//         MPI_LAND,                             // operation
//         this->underlying().mpi_communicator() // communicator
//     );
//     return every_buffer_large_enough;
// }
