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
/// - \ref kamping::send_recv_buf() containing the data that is sent to the other ranks.
/// The following parameters are optional:
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c
/// Communicator is used, see root().
/// - \ref kamping::send_recv_count() specifying how many elements are broadcasted. If not specified, will be
/// communicated thorugh an additional bcast. If specified, has to be the same on all ranks (including the root). Has to
/// either be specified or not specified on all ranks.
/// @todo Add support for `bcast<int>(..)` style deduction of send_recv_buf's type on non-root ranks.
/// @todo Add support for unnamed first parameter send_recv_buf.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <typename... Args>
auto kamping::Communicator::bcast(Args... args) const {
    using namespace ::kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args, KAMPING_REQUIRED_PARAMETERS(send_recv_buf), KAMPING_OPTIONAL_PARAMETERS(root, send_recv_count));

    // Get the root PE
    auto&& root = select_parameter_type_or_default<ParameterType::root, Root>(std::tuple(this->root()), args...);
    KASSERT(this->is_valid_rank(root.rank()), "Invalid rank as root.", assert::light);

    // Get the send_recv_buf
    // For now, the user *has* to provide a send recv buf
    // using default_send_recv_buf_type =
    // decltype(kamping::send_recv_buf(NewContainer<std::vector<send_value_type>>{}));

    // const bool  has_user_provided_send_recv_buf = has_parameter_type<ParameterType::send_recv_buf>(args...);
    // auto&& send_recv_buf =
    //     internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_send_recv_buf_type>(
    //         std::tuple(), args...);
    auto&& send_recv_buf = internal::select_parameter_type<internal::ParameterType::send_recv_buf>(args...);
    using value_type     = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;
    auto mpi_value_type  = mpi_datatype<value_type>();

    // Get the recv_count
    constexpr bool has_user_provided_send_recv_count = has_parameter_type<ParameterType::send_recv_count, Args...>();

    // If I'm the root, assert, that I have a send_recv_buf which is not empty.
    if (this->is_root(root.rank())) {
        /// @todo Uncomment, once the send_recv_buf is optional.
        // KASSERT(has_user_provided_send_recv_buf, "The send_recv_buf is mandatory at the root.", assert::light);
        KASSERT(send_recv_buf.size() > 0u, "The send_recv_buf on the root rank must not be empty.");
    }

    // Assume that either all ranks have send_recv_count or none of them hast -> need to broadcast the amount of data to
    // transfer.
    KASSERT(
        this->is_same_on_all_ranks(has_user_provided_send_recv_count),
        "The send_recv_count must be either provided on all ranks or on no rank.", assert::light_communication);
    size_t send_recv_count = 0;
    if constexpr (has_user_provided_send_recv_count) {
        send_recv_count =
            asserting_cast<size_t>(*(select_parameter_type<ParameterType::send_recv_count>(args...).get().data()));
    } else {
        if (this->is_root(root.rank())) {
            send_recv_count = send_recv_buf.size();
        }

        // Transfer the send_recv_count
        // This error code is unused if KTHROW is removed at compile time.
        [[maybe_unused]] int err = MPI_Bcast(
            &send_recv_count,                          // buffer*
            1,                                         // count
            mpi_datatype<decltype(send_recv_count)>(), // datatype
            root.rank_signed(),                        // root
            this->mpi_communicator()                   // MPI_Comm comm
        );
        THROW_IF_MPI_ERROR(err, MPI_Bcast);

        // If I'm not the root, resize my send_recv_buf to be able to hold all received data.
        if (!this->is_root(root.rank())) {
            send_recv_buf.resize(send_recv_count);
        }
        KASSERT(send_recv_buf.size() == send_recv_count, assert::light);
    }
    KASSERT(
        this->is_same_on_all_ranks(send_recv_count), "The send_recv_count must be equal on all ranks.",
        assert::light_communication);

    /// @todo Implement and test that passing a const-buffer is allowed on the root process but not on all other
    /// processes.

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
