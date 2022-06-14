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
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(root, send_recv_buf, send_recv_count));

    static_assert(
        all_parameters_are_rvalues<Args...>,
        "All parameters have to be passed in as rvalue references, meaning that you must not hold a variable "
        "returned by the named parameter helper functions like recv_buf().");

    // Get the root PE
    auto&& root = select_parameter_type_or_default<ParameterType::root, Root>(std::tuple(this->root()), args...);
    KASSERT(this->is_valid_rank(root.rank()), "Invalid rank as root.", assert::light);

    // Get the send_recv_buf
    using default_send_recv_buf_type = decltype(kamping::send_recv_buf(NewContainer<std::vector<send_value_type>>{}));

    const bool  has_user_provided_send_recv_buf = has_parameter_type<ParameterType::send_recv_buf>(args...);
    auto&& send_recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_send_recv_buf_type>(
            std::tuple(), args...);
    using value_type                            = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;
    auto mpi_value_type                         = mpi_datatype<value_type>();

    // Get the recv_count
    const bool has_user_provided_send_recv_count = has_parameter_type<ParameterType::send_recv_count>(args...);

    // If I'm the root, assert, that I have a send_recv_buf which is not empty.
    if (this->is_root(root.rank())) {
        KASSERT(has_user_provided_send_recv_buf, "The send_recv_buf is mandatory at the root.", assert::light);
        KASSERT(send_recv_buf.size() > 0, "The send_recv_buf on the root rank must not be empty.");
    }

    // Assume that either all ranks have send_recv_count or none of them hast -> need to broadcast the amount of data to
    // transfer.
    // TODO Assert that either all ranks or no rank has a send_recv_count.
    size_t send_recv_count = 0;
    if (has_user_provided_send_recv_count) {
        send_recv_count = select_parameter_type<ParameterType::send_recv_count>(args...).get();
    } else {
        if (this->is_root(root.rank())) {
            send_recv_count = send_recv_buf.size();
        }
        // This error code is unused if KTHROW is removed at compile time.
        [[maybe_unused]] int err = MPI_Bcast(
            &send_recv_count,                          // buffer*
            asserting_cast<int>(send_recv_buf.size()), // count
            mpi_value_type,                            // datatype
            root.rank_signed(),                        // root
            this->mpi_communicator()                   // MPI_Comm comm
        );
        THROW_IF_MPI_ERROR(err, MPI_Bcast);

        // If I'm not the root, resize my send_recv_buf to be able to hold all received data.
        if (!this->is_root(root.rank())) {
            send_recv_buf.resize(send_recv_count);
        }
    }
    // TODO Assert, that the recv_counts are the same on all ranks

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
        std::move(send_recv_buf), BufferCategoryNotUsed{}, BufferCategoryNotUsed{}, BufferCategoryNotUsed{},
        BufferCategoryNotUsed{});
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
