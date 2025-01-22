// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
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

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/collectives_helpers.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @addtogroup kamping_collectives
/// @{

/// @brief Wrapper for \c MPI_Bcast
///
/// This wrapper for \c MPI_Bcast sends data from the root to all other ranks.
///
/// The following buffer is required on the root rank:
/// - \ref kamping::send_recv_buf() containing the data that is sent to the other ranks. Non-root ranks must allocate
/// and provide this buffer or provide the receive type as a template parameter to \c bcast() as
/// it's used for deducing the value type. The buffer will be resized on non-root ranks according to the buffer's
/// kamping::BufferResizePolicy.
///
/// The following parameter is optional but causes additional communication if not present.
/// - \ref kamping::send_recv_count() specifying how many elements are broadcasted. This parameter must be given either
/// on all or none of the ranks. If not specified, the count is set to the size of kamping::send_recv_buf() on
/// root and broadcasted to all other ranks. This parameter is mandatory if \ref kamping::send_recv_type() is given.
///
/// The following parameter are optional:
/// - \ref kamping::send_recv_type() specifying the \c MPI datatype to use as send type on the root PE and recv type on
/// all non-root PEs. If omitted, the \c MPI datatype is derived automatically based on send_recv_buf's underlying \c
/// value_type.
///
/// - \ref kamping::root() specifying an alternative root. If not present, the default root of the \c
/// Communicator is used, see root().
///
/// @tparam recv_value_type_tparam The type that is received. Only required when no \ref kamping::send_recv_buf() is
/// given.
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional parameters described above.
/// @return Result object wrapping the output parameters to be returned by value.
///
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
/// <hr>
/// \include{doc} docs/resize_policy.dox
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::bcast(Args... args) const {
    using namespace ::kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(),
        KAMPING_OPTIONAL_PARAMETERS(send_recv_buf, root, send_recv_count, send_recv_type)
    );

    // Get the root PE
    auto&& root = select_parameter_type_or_default<ParameterType::root, internal::RootDataBuffer>(
        std::tuple(this->root()),
        args...
    );
    KASSERT(this->is_valid_rank(root.rank_signed()), "Invalid rank as root.", assert::light);
    KASSERT(
        is_same_on_all_ranks(root.rank_signed()),
        "root() parameter must be the same on all ranks.",
        assert::light_communication
    );

    using default_send_recv_buf_type =
        decltype(kamping::send_recv_buf(alloc_new<DefaultContainerType<recv_value_type_tparam>>));
    auto send_recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::send_recv_buf, default_send_recv_buf_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType, serialization_support_tag>();
    constexpr bool is_serialization_used = internal::buffer_uses_serialization<decltype(send_recv_buf)>;
    if constexpr (is_serialization_used) {
        KAMPING_UNSUPPORTED_PARAMETER(Args, send_recv_count, when using serialization);
        KAMPING_UNSUPPORTED_PARAMETER(Args, send_recv_type, when using serialization);
        if (this->is_root(root.rank_signed())) {
            send_recv_buf.underlying().serialize();
        }
    }

    using value_type = typename std::remove_reference_t<decltype(send_recv_buf)>::value_type;
    static_assert(
        !std::is_same_v<value_type, internal::unused_tparam>,
        "No send_recv_buf parameter provided and no receive value given as template parameter. One of these is "
        "required."
    );

    constexpr bool buffer_is_modifiable = std::remove_reference_t<decltype(send_recv_buf)>::is_modifiable;

    auto send_recv_type = determine_mpi_send_recv_datatype<value_type, decltype(send_recv_buf)>(args...);
    [[maybe_unused]] constexpr bool send_recv_type_is_in_param = !has_to_be_computed<decltype(send_recv_type)>;

    KASSERT(
        this->is_root(root.rank_signed()) || buffer_is_modifiable,
        "send_recv_buf must be modifiable on all non-root ranks.",
        assert::light
    );

    // Get the optional recv_count parameter. If the parameter is not given, allocate a new container.
    using default_count_type = decltype(kamping::send_recv_count_out());
    auto count_param = internal::select_parameter_type_or_default<ParameterType::send_recv_count, default_count_type>(
                           std::tuple(),
                           args...
    )
                           .construct_buffer_or_rebind();

    constexpr bool count_has_to_be_computed = has_to_be_computed<decltype(count_param)>;
    KASSERT(
        is_same_on_all_ranks(count_has_to_be_computed),
        "send_recv_count() parameter is either deduced on all ranks or must be explicitly provided on all ranks.",
        assert::light_communication
    );
    if constexpr (count_has_to_be_computed) {
        int       count;
        int const NO_BUF_ON_ROOT = -1;
        if (this->is_root(root.rank_signed())) {
            count_param.underlying() = asserting_cast<int>(send_recv_buf.size());
            count                    = count_param.get_single_element();

            if constexpr (!has_parameter_type<internal::ParameterType::send_recv_buf, Args...>()) {
                // if no send_recv_buf is provided on the root rank, we abuse the recv_count parameter to signal that
                // there is no buffer on the root rank to all other ranks.
                // This allows us to fail on all ranks if the root rank does not provide a buffer.
                count = NO_BUF_ON_ROOT;
            };
        }
        // Transfer the recv_count
        // This error code is unused if KTHROW is removed at compile time.
        /// @todo Use bcast_single for this.
        [[maybe_unused]] int err = MPI_Bcast(
            &count,                          // buffer
            1,                               // count
            mpi_datatype<decltype(count)>(), // datatype
            root.rank_signed(),              // root
            this->mpi_communicator()         // comm
        );
        this->mpi_error_hook(err, "MPI_Bcast");

        // it is valid to do this check here, because if no send_recv_buf is provided on the root rank, we have
        // always have deduce counts and get into this branch.
        KASSERT(count != NO_BUF_ON_ROOT, "send_recv_buf must be provided on the root rank.", assert::light);

        // Output the recv count via the output_parameter
        count_param.underlying() = count;
    } else {
        KASSERT(
            (!this->is_root(root.rank_signed()) || has_parameter_type<internal::ParameterType::send_recv_buf, Args...>()
            ),
            "send_recv_buf must be provided on the root rank.",
            assert::light
        );
    }

    // Resize my send_recv_buf to be able to hold all received data on all non_root ranks.
    // Trying to resize a single element buffer to something other than 1 will throw an error.
    if (!this->is_root(root.rank_signed())) {
        auto compute_recv_buffer_size = [&] {
            return asserting_cast<size_t>(count_param.get_single_element());
        };
        send_recv_buf.resize_if_requested(compute_recv_buffer_size);
        KASSERT(
            // if the send_recv type is user provided, kamping cannot make any assumptions about the required size of
            // the send_recv buffer
            send_recv_type_is_in_param || send_recv_buf.size() >= compute_recv_buffer_size(),
            "send/receive buffer is not large enough to hold all received elements on a non-root rank.",
            assert::light
        );
    }

    // Perform the broadcast. The error code is unused if KTHROW is removed at compile time.
    [[maybe_unused]] int err = MPI_Bcast(
        send_recv_buf.data(),                // buffer
        count_param.get_single_element(),    // count
        send_recv_type.get_single_element(), // datatype
        root.rank_signed(),                  // root
        this->mpi_communicator()             // comm
    );
    this->mpi_error_hook(err, "MPI_Bcast");

    return make_mpi_result<std::tuple<Args...>>(
        deserialization_repack<is_serialization_used>(std::move(send_recv_buf)),
        std::move(count_param),
        std::move(send_recv_type)
    );
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
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return The single broadcasted value.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::bcast_single(Args... args) const {
    //! If you expand this function to not being only a simple wrapper around bcast, you have to write more unit tests!

    using namespace kamping::internal;

    // In contrast to bcast(...), send_recv_count is not a possible parameter.
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(send_recv_buf, root));

    // Get the root PE
    auto&& root = select_parameter_type_or_default<ParameterType::root, internal::RootDataBuffer>(
        std::tuple(this->root()),
        args...
    );
    // we have to do this check with communication, because otherwise the other ranks would already start with the
    // broadcast and indefinitely wait for the root
    if constexpr (kassert::internal::assertion_enabled(assert::light_communication)) {
        bool root_has_buffer = has_parameter_type<internal::ParameterType::send_recv_buf, Args...>();
        int  err = MPI_Bcast(&root_has_buffer, 1, MPI_CXX_BOOL, root.rank_signed(), this->mpi_communicator());
        this->mpi_error_hook(err, "MPI_Bcast");
        KASSERT(root_has_buffer, "send_recv_buf must be provided on the root rank.", assert::light_communication);
    }

    if constexpr (has_parameter_type<ParameterType::send_recv_buf, Args...>()) {
        using send_recv_buf_type = buffer_type_with_requested_parameter_type<ParameterType::send_recv_buf, Args...>;
        static_assert(
            send_recv_buf_type::is_single_element,
            "The underlying container has to be a single element \"container\""
        );
        return this->bcast<recv_value_type_tparam>(std::forward<Args>(args)..., send_recv_count(1));
    } else {
        return *this->bcast<recv_value_type_tparam>(std::forward<Args>(args)..., send_recv_count(1)).data();
    }
}
/// @}
