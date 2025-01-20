// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/implementation_helpers.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/helpers.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"

///// @addtogroup kamping_p2p
/// @{

// @brief Wrapper for \c MPI_Isend.
///
/// This wraps \c MPI_Isend. This operation sends the elements in the input buffer provided via \c
/// kamping::send_buf() to the specified receiver rank using standard send mode without blocking. The call is associated
/// with a \ref kamping::Request (either allocated by KaMPIng or provided by the user). Before accessing the result the
/// user has to complete the request.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent.
///
/// - \ref kamping::destination() the receiving rank.
///
/// The following parameters are optional:
/// - \ref kamping::send_count() specifying how many elements of the buffer are sent.
/// If omitted, the size of the send buffer is used as a default. This parameter is mandatory if \ref
/// kamping::send_type() is given.
///
///  - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype is
/// derived automatically based on send_buf's underlying \c value_type.
///
/// - \ref kamping::tag() the tag added to the message. Defaults to the communicator's default tag (\ref
/// Communicator::default_tag()) if not present.
///
/// - \ref kamping::request() The request object to associate this operation with. Defaults to a library allocated
/// request object, which can be access via the returned result.
///
/// - \ref kamping::send_mode() the send mode to use. Defaults to standard MPI_Send.
///
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
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::isend(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, destination),
        KAMPING_OPTIONAL_PARAMETERS(send_count, tag, send_mode, request, send_type)
    );

    auto send_buf =
        internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;

    auto send_type = internal::determine_mpi_send_datatype<send_value_type>(args...);

    using default_send_count_type = decltype(kamping::send_count_out());
    auto send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_send_count_type>(
            {},
            args...
        )
            .construct_buffer_or_rebind();
    if constexpr (has_to_be_computed<decltype(send_count)>) {
        send_count.underlying() = asserting_cast<int>(send_buf.size());
    }

    auto const&    destination = internal::select_parameter_type<internal::ParameterType::destination>(args...);
    constexpr auto rank_type   = std::remove_reference_t<decltype(destination)>::rank_type;
    static_assert(
        rank_type == RankType::value || rank_type == RankType::null,
        "Please provide an explicit destination or destination(ranks::null)."
    );
    using default_request_param = decltype(kamping::request());
    auto&& request_param =
        internal::select_parameter_type_or_default<internal::ParameterType::request, default_request_param>(
            std::tuple{},
            args...
        );

    using default_tag_buf_type = decltype(kamping::tag(0));

    auto&& tag_param = internal::select_parameter_type_or_default<internal::ParameterType::tag, default_tag_buf_type>(
        std::tuple(this->default_tag()),
        args...
    );

    // this ensures that the user does not try to pass MPI_ANY_TAG, which is not allowed for sends
    static_assert(
        std::remove_reference_t<decltype(tag_param)>::tag_type == TagType::value,
        "Please provide a tag for the message."
    );
    int tag = tag_param.tag();
    KASSERT(
        Environment<>::is_valid_tag(tag),
        "invalid tag " << tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
    );

    using send_mode_obj_type = decltype(internal::select_parameter_type_or_default<
                                        internal::ParameterType::send_mode,
                                        internal::SendModeParameter<internal::standard_mode_t>>(std::tuple(), args...));
    using send_mode          = typename std::remove_reference_t<send_mode_obj_type>::send_mode;

    // RankType::null is valid, RankType::any is not.
    KASSERT(is_valid_rank_in_comm(destination, *this, true, false), "Invalid destination rank.");

    // store all parameters/buffers for which we have to ensure pointer stability until completion of the immediate MPI
    // call on the heap.
    auto buffers_on_heap = move_buffer_to_heap(std::move(send_buf));

    auto* send_buf_ptr =
        internal::select_parameter_type_in_tuple<internal::ParameterType::send_buf>(*buffers_on_heap).data();

    if constexpr (std::is_same_v<send_mode, internal::standard_mode_t>) {
        [[maybe_unused]] int err = MPI_Isend(
            send_buf_ptr,                            // send_buf
            send_count.get_single_element(),         // send_count
            send_type.get_single_element(),          // send_type
            destination.rank_signed(),               // destination
            tag,                                     // tag
            this->mpi_communicator(),                // comm
            request_param.underlying().request_ptr() // request
        );
        this->mpi_error_hook(err, "MPI_Isend");
    } else if constexpr (std::is_same_v<send_mode, internal::buffered_mode_t>) {
        [[maybe_unused]] int err = MPI_Ibsend(
            send_buf_ptr,                            // send_buf
            send_count.get_single_element(),         // send_count
            send_type.get_single_element(),          // send_type
            destination.rank_signed(),               // destination
            tag,                                     // tag
            this->mpi_communicator(),                // comm
            request_param.underlying().request_ptr() // request
        );
        this->mpi_error_hook(err, "MPI_Ibsend");
    } else if constexpr (std::is_same_v<send_mode, internal::synchronous_mode_t>) {
        [[maybe_unused]] int err = MPI_Issend(
            send_buf_ptr,                            // send_buf
            send_count.get_single_element(),         // send_count
            send_type.get_single_element(),          // send_type
            destination.rank_signed(),               // destination
            tag,                                     // tag
            this->mpi_communicator(),                // comm
            request_param.underlying().request_ptr() // request
        );
        this->mpi_error_hook(err, "MPI_Issend");
    } else if constexpr (std::is_same_v<send_mode, internal::ready_mode_t>) {
        [[maybe_unused]] int err = MPI_Irsend(
            send_buf_ptr,                            // send_buf
            send_count.get_single_element(),         // send_count
            send_type.get_single_element(),          // send_type
            destination.rank_signed(),               // destination
            tag,                                     // tag
            this->mpi_communicator(),                // comm
            request_param.underlying().request_ptr() // request
        );
        this->mpi_error_hook(err, "MPI_Irsend");
    }
    return internal::make_nonblocking_result<std::tuple<Args...>>(std::move(request_param), std::move(buffers_on_heap));
}

/// @brief Convenience wrapper for MPI_Ibsend. Calls \ref kamping::Communicator::isend() with the appropriate send mode
/// set.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::ibsend(Args... args) const {
    return this->isend(std::forward<Args>(args)..., send_mode(send_modes::buffered));
}

/// @brief Convenience wrapper for MPI_Issend. Calls \ref kamping::Communicator::isend() with the appropriate send mode
/// set.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::issend(Args... args) const {
    return this->isend(std::forward<Args>(args)..., send_mode(send_modes::synchronous));
}

/// @brief Convenience wrapper for MPI_Irsend. Calls \ref kamping::Communicator::isend() with the appropriate send mode
/// set.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::irsend(Args... args) const {
    return this->isend(std::forward<Args>(args)..., send_mode(send_modes::ready));
}
/// @}
