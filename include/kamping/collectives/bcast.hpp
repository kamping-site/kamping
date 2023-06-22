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

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_function_wrapper_helpers.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"

/// @brief Wrapper for \c MPI_Bcast
///
/// This wrapper for \c MPI_Bcast sends data from the root to all other ranks.
///
/// The following buffer is required on the root rank:
/// - \ref kamping::send_recv_buf() containing the data that is sent to the other ranks. Non-root ranks must allocate
/// and provide this buffer or provide the receive type as a template parameter to \c bcast() as
/// it's used for deducing the value type. The container will be resized on
/// non-root ranks to fit exactly the received data.
///
/// The following parameter is optional but causes additional communication if not present.
/// - \ref kamping::recv_counts() specifying how many elements are broadcasted. If not specified, will be
/// communicated through an additional bcast. If not specified, we broadcast the whole send_recv_buf. If specified,
/// has to be the same on all ranks (including the root). Has to either be specified or not specified on all ranks.
///
/// The following parameter is optional:
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c
/// Communicator is used, see root().
///
/// @todo Add support for unnamed first parameter send_recv_buf.
///
/// @tparam recv_value_type_tparam The type that is received. Only required when no \ref kamping::send_recv_buf() is
/// given.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output buffer if not specified as input parameter.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::bcast(Args... args) const {
    using namespace ::kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(),
        KAMPING_OPTIONAL_PARAMETERS(send_recv_buf, root, recv_counts)
    );

    // Get the root PE
    auto&& root = select_parameter_type_or_default<ParameterType::root, internal::RootDataBuffer>(
        std::tuple(this->root()),
        args...
    );
    KASSERT(this->is_valid_rank(root.rank_signed()), "Invalid rank as root.", assert::light);

    if (this->is_root(root.rank_signed())) {
        KASSERT(
            has_parameter_type<internal::ParameterType::send_recv_buf>(args...),
            "send_recv_buf must be provided on the root rank.",
            assert::light
        );
    }

    using default_send_recv_buf_type =
        decltype(kamping::send_recv_buf(alloc_new<DefaultContainerType<recv_value_type_tparam>>));
    auto&& send_recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::send_recv_buf, default_send_recv_buf_type>(
            std::tuple(),
            args...
        );

    using value_type = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;
    static_assert(!std::is_const_v<decltype(send_recv_buf)>, "Const send_recv_buffers are not allowed.");
    static_assert(
        !std::is_same_v<value_type, internal::unused_tparam>,
        "No send_recv_buf parameter provided and no receive value given as template parameter. One of these is "
        "required."
    );

    auto mpi_value_type = mpi_datatype<value_type>();

    // Get the optional recv_count parameter. If the parameter is not given, allocate a new container.
    using default_recv_count_type = decltype(kamping::recv_counts_out(alloc_new<int>));
    auto&& recv_count_param =
        internal::select_parameter_type_or_default<ParameterType::recv_counts, default_recv_count_type>(
            std::tuple(),
            args...
        );

    constexpr bool recv_count_is_output_parameter = has_to_be_computed<decltype(recv_count_param)>;
    KASSERT(
        is_same_on_all_ranks(recv_count_is_output_parameter),
        "recv_count() parameter is an output parameter on some PEs, but not on alle PEs.",
        assert::light_communication
    );

    // If it is not user provided, broadcast the size of send_recv_buf from the root to all ranks.
    static_assert(
        std::remove_reference_t<decltype(recv_count_param)>::is_single_element,
        "recv_counts() parameter must be a single value."
    );
    int recv_count = recv_count_param.get_single_element();
    if constexpr (recv_count_is_output_parameter) {
        if (this->is_root(root.rank_signed())) {
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
    if (this->is_root(root.rank_signed())) {
        KASSERT(
            asserting_cast<size_t>(recv_count) == send_recv_buf.size(),
            "If a recv_count() is provided on the root rank, it has to be equal to the number of elements in the "
            "send_recv_buf. For partial transfers, use a kamping::Span."
        );
    }
    KASSERT(
        this->is_same_on_all_ranks(recv_count),
        "The recv_count must be equal on all ranks.",
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

    return make_mpi_result(std::move(send_recv_buf), std::move(recv_count_param));
} // namespace kamping::internal

/// @brief Wrapper for \c MPI_Bcast
///
/// This wrapper for \c MPI_Bcast sends a single value from the root to all other ranks. Calling \c bcast_single() is a
/// shorthand for calling `bcast(..., recv_counts(1))`. It always issues only a single \c MPI_Bcast call, as no receive
/// counts have to be exchanged.
///
/// The following buffer is required on the root rank:
/// - \ref kamping::send_recv_buf() containing the single value that is sent to the other ranks. Non-root ranks must
/// either allocate and provide this buffer or provide the receive type as a template parameter to \c bcast_single() as
/// it's used for deducing the value type.
///
/// The following parameter is optional:
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c Communicator is
/// used, see root().
///
/// @todo Add support for unnamed first parameter send_recv_buf.
///
/// @tparam recv_value_type_tparam The type that is received. Only required when no \ref kamping::send_recv_buf() is
/// given.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return The single broadcasted value.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::bcast_single(Args... args) const {
    //! If you expand this function to not being only a simple wrapper around bcast, you have to write more unit tests!

    using namespace kamping::internal;

    // In contrast to bcast(...), the recv_counts is not a possible parameter.
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(send_recv_buf, root));

    // Get the root PE
    auto&& root = select_parameter_type_or_default<ParameterType::root, internal::RootDataBuffer>(
        std::tuple(this->root()),
        args...
    );
    KASSERT(this->is_valid_rank(root.rank_signed()), "Invalid rank as root.", assert::light);

    if (this->is_root(root.rank_signed())) {
        KASSERT(
            has_parameter_type<internal::ParameterType::send_recv_buf>(args...),
            "send_recv_buf must be provided on the root rank.",
            assert::light
        );
    }

    if constexpr (has_parameter_type<internal::ParameterType::send_recv_buf, Args...>()) {
        KASSERT(
            select_parameter_type<ParameterType::send_recv_buf>(args...).size() == 1u,
            "The send/receive buffer has to be of size 1 on all ranks.",
            assert::light
        );
    }

    if constexpr (has_parameter_type<internal::ParameterType::send_recv_buf, Args...>()) {
        return this->bcast<recv_value_type_tparam>(std::forward<Args>(args)..., recv_counts(1));
    } else {
        return this->bcast<recv_value_type_tparam>(std::forward<Args>(args)..., recv_counts(1))
            .extract_recv_buffer()[0];
    }
}
