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
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
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
/// The following buffer is required:
/// - \ref kamping::send_recv_buf() containing the data that is sent to the other ranks. Non-root ranks must allocate
/// and provide this buffer as it's needed for deducing the value type. The container will be resized on non-root ranks
/// to fit exactly the received data.
/// The following parameter is optional but causes additional communication if not present.
/// - \ref kamping::recv_count() specifying how many elements are broadcasted. If not specified, will be
/// communicated through an additional bcast. If not specified, we broadcast the whole send_recv_buf. If specified,
/// has to be the same on all ranks (including the root). Has to either be specified or not specified on all ranks. The
/// following parameter is optional:
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c
/// Communicator is used, see root().
/// @todo Add support for `bcast<int>(..)` style deduction of send_recv_buf's type on non-root ranks.
/// @todo Add support for unnamed first parameter send_recv_buf.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <typename... Args>
auto kamping::Communicator::bcast(Args... args) const {
    using namespace ::kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args, KAMPING_REQUIRED_PARAMETERS(send_recv_buf), KAMPING_OPTIONAL_PARAMETERS(root, recv_counts)
    );

    // Get the root PE
    auto&& root = select_parameter_type_or_default<ParameterType::root, Root>(std::tuple(this->root()), args...);
    KASSERT(this->is_valid_rank(root.rank()), "Invalid rank as root.", assert::light);

    // Get the send_recv_buf; for now, the user *has* to provide a send-receive buffer.
    auto&& send_recv_buf = internal::select_parameter_type<internal::ParameterType::send_recv_buf>(args...);
    using value_type     = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;
    static_assert(!std::is_const_v<decltype(send_recv_buf)>, "Const send_recv_buf'fers are not allowed.");
    auto mpi_value_type = mpi_datatype<value_type>();

    /// @todo Uncomment, once the send_recv_buf is optional.
    // if (this->is_root(root.rank())) {
    //     KASSERT(has_user_provided_send_recv_buf, "The send_recv_buf is mandatory at the root.", assert::light);
    // }

    // Get the optional recv_count parameter. If the parameter is not given, allocate a new container.
    using default_recv_count_type = decltype(kamping::recv_count_out(NewContainer<int>{}));
    auto&& recv_count_param =
        internal::select_parameter_type_or_default<ParameterType::recv_counts, default_recv_count_type>(
            std::tuple(), args...
        );

    constexpr bool recv_count_is_output_parameter = has_to_be_computed<decltype(recv_count_param)>;
    KASSERT(
        is_same_on_all_ranks(recv_count_is_output_parameter),
        "recv_count() parameter is an output parameter on some PEs, but not on alle PEs.", assert::light_communication
    );

    // If it is not user provided, broadcast the size of send_recv_buf from the root to all ranks.
    int recv_count = recv_count_param.get_single_element();
    if constexpr (recv_count_is_output_parameter) {
        if (this->is_root(root.rank())) {
            recv_count = asserting_cast<int>(send_recv_buf.size());
        }
        // Transfer the recv_count
        // This error code is unused if KTHROW is removed at compile time.
        /// @todo Use bcast_single for this.
        [[maybe_unused]] int err = MPI_Bcast(
            &recv_count,                          // buffer
            1,                                    // count
            mpi_datatype<decltype(recv_count)>(), // datatype
            root.rank_signed(),                   // root
            this->mpi_communicator()              // comm
        );
        THROW_IF_MPI_ERROR(err, MPI_Bcast);

        // Output the recv count via the output_parameter
        *recv_count_param.data() = recv_count;
    }
    if (this->is_root(root.rank())) {
        KASSERT(
            asserting_cast<size_t>(recv_count) == send_recv_buf.size(),
            "If a recv_count() is provided on the root rank, it has to be equal to the number of elements in the "
            "send_recv_buf. For partial transfers, use a kamping::Span."
        );
    }
    KASSERT(
        this->is_same_on_all_ranks(recv_count), "The recv_count must be equal on all ranks.",
        assert::light_communication
    );

    // Resize my send_recv_buf to be able to hold all received data.
    // Trying to resize a single element buffer to something other than 1 will throw an error.
    send_recv_buf.resize(asserting_cast<size_t>(recv_count));

    // Perform the broadcast. The error code is unused if KTHROW is removed at compile time.
    [[maybe_unused]] int err = MPI_Bcast(
        send_recv_buf.data(),                      // buffer
        asserting_cast<int>(send_recv_buf.size()), // count
        mpi_value_type,                            // datatype
        root.rank_signed(),                        // root
        this->mpi_communicator()                   // comm
    );
    THROW_IF_MPI_ERROR(err, MPI_Bcast);

    return MPIResult(
        std::move(send_recv_buf), std::move(recv_count_param), BufferCategoryNotUsed{}, BufferCategoryNotUsed{}
    );
} // namespace kamping::internal

template <typename... Args>
auto kamping::Communicator::bcast_single(Args... args) const {
    //! If your expand this function to not being only a simple wrapper arount bcast, you have to write more unit tests!
    // In contrast to bcast(...), the recv_count is not a possible parameter.
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_recv_buf), KAMPING_OPTIONAL_PARAMETERS(root));

    return this->bcast(std::forward<Args>(args)..., recv_counts(1));
}
