// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
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
#include "kamping/data_buffer.hpp"
#include "kamping/implementation_helpers.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/request.hpp"
#include "kamping/result.hpp"

/// @brief Wrapper for \c MPI_Isend.
///
/// This wraps \c MPI_Isend. This operation sends the elements in the input buffer provided via \c
/// kamping::send_buf() to the specified receiver rank using standard send mode without blocking. The call is associated
/// with a \ref kamping::Request (either allocated by KaMPIng or provided by the user). Before accessing the result the
/// user has to complete the request.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent.
/// - \ref kamping::destination() the receiving rank.
///
/// The following parameters are optional:
/// - \ref kamping::tag() the tag added to the message. Defaults to the communicator's default tag (\ref
/// Communicator::default_tag()) if not present.
/// - \ref kamping::send_mode() the send mode to use. Defaults to standard MPI_Send.
/// - \ref kamping::request() The request object to associate this operation with. Defaults to a library allocated
/// request object, which can be access via the returned result.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described above.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::isend(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, destination),
        KAMPING_OPTIONAL_PARAMETERS(tag, send_mode, request)
    );

    auto& send_buf_param  = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
    auto  send_buf        = send_buf_param.get();
    using send_value_type = typename std::remove_reference_t<decltype(send_buf_param)>::value_type;

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
        "invalid tag " << tag << ", maximum allowed tag is " << Environment<>::tag_upper_bound()
    );

    using send_mode_obj_type = decltype(internal::select_parameter_type_or_default<
                                        internal::ParameterType::send_mode,
                                        internal::SendModeParameter<internal::standard_mode_t>>(std::tuple(), args...));
    using send_mode          = typename std::remove_reference_t<send_mode_obj_type>::send_mode;

    auto mpi_send_type = mpi_datatype<send_value_type>();

    // RankType::null is valid, RankType::any is not.
    KASSERT(is_valid_rank_in_comm(destination, *this, true, false), "Invalid destination rank.");

    if constexpr (std::is_same_v<send_mode, internal::standard_mode_t>) {
        [[maybe_unused]] int err = MPI_Isend(
            send_buf.data(),                          // send_buf
            asserting_cast<int>(send_buf.size()),     // send_count
            mpi_send_type,                            // send_type
            destination.rank_signed(),                // destination
            tag,                                      // tag
            this->mpi_communicator(),                 // comm
            &request_param.underlying().mpi_request() // request
        );
        THROW_IF_MPI_ERROR(err, MPI_Isend);
    } else if constexpr (std::is_same_v<send_mode, internal::buffered_mode_t>) {
        [[maybe_unused]] int err = MPI_Ibsend(
            send_buf.data(),                          // send_buf
            asserting_cast<int>(send_buf.size()),     // send_count
            mpi_send_type,                            // send_type
            destination.rank_signed(),                // destination
            tag,                                      // tag
            this->mpi_communicator(),                 // comm
            &request_param.underlying().mpi_request() // request
        );
        THROW_IF_MPI_ERROR(err, MPI_Ibsend);
    } else if constexpr (std::is_same_v<send_mode, internal::synchronous_mode_t>) {
        [[maybe_unused]] int err = MPI_Issend(
            send_buf.data(),                          // send_buf
            asserting_cast<int>(send_buf.size()),     // send_count
            mpi_send_type,                            // send_type
            destination.rank_signed(),                // destination
            tag,                                      // tag
            this->mpi_communicator(),                 // comm
            &request_param.underlying().mpi_request() // request
        );
        THROW_IF_MPI_ERROR(err, MPI_Issend);
    } else if constexpr (std::is_same_v<send_mode, internal::ready_mode_t>) {
        [[maybe_unused]] int err = MPI_Irsend(
            send_buf.data(),                          // send_buf
            asserting_cast<int>(send_buf.size()),     // send_count
            mpi_send_type,                            // send_type
            destination.rank_signed(),                // destination
            tag,                                      // tag
            this->mpi_communicator(),                 // comm
            &request_param.underlying().mpi_request() // request
        );
        THROW_IF_MPI_ERROR(err, MPI_Irsend);
    }
    return make_nonblocking_result(std::move(request_param));
}

/// @brief Convenience wrapper for MPI_Ibsend. Calls \ref kamping::Communicator::isend() with the appropriate send mode
/// set.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::ibsend(Args... args) const {
    return this->isend(std::forward<Args>(args)..., send_mode(send_modes::buffered));
}

/// @brief Convenience wrapper for MPI_Issend. Calls \ref kamping::Communicator::isend() with the appropriate send mode
/// set.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::issend(Args... args) const {
    return this->isend(std::forward<Args>(args)..., send_mode(send_modes::synchronous));
}

/// @brief Convenience wrapper for MPI_Irsend. Calls \ref kamping::Communicator::isend() with the appropriate send mode
/// set.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::irsend(Args... args) const {
    return this->isend(std::forward<Args>(args)..., send_mode(send_modes::ready));
}