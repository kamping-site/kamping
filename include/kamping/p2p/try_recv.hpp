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
#include <utility>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/implementation_helpers.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/probe.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"
#include "kamping/status.hpp"

/// @brief Wrapper for \c MPI_Improbe and MPI_Mrecv. Receives a message if one is available.
///
/// In contrast to \ref kamping::Communicator::recv(), this method does not block if no message is available. Instead,
/// it will return a empty \c std::optional
///
/// The following parameters are optional:
/// - \ref kamping::recv_buf() the buffer to receive the message into. If possible, this buffer will be resized to
/// accommodate the number of elements to receive. Use \c kamping::Span with enough space if you do not want the buffer
/// to be resized. If no \ref kamping::recv_buf() is provided, the type that should be received has to be passed as a
/// template parameter to \c try_recv().
/// - \ref kamping::tag() the tag of the received message. Defaults to receiving for an arbitrary tag, i.e. \c
/// tag(tags::any).
/// - \ref kamping::source() the source rank of the message to receive. Defaults to probing for an arbitrary source,
/// i.e. \c source(rank::any).
/// - \ref kamping::status() or \ref kamping::status_out(). Returns info about the received message by setting the
/// appropriate fields in the status object passed by the user. If \ref kamping::status_out() is passed, constructs a
/// status object which may be retrieved by the user. The status can be ignored by passing \c
/// kamping::status(kamping::ignore<>). This is the default.
///
/// @tparam recv_value_type_tparam The type that is received. Only required when no \ref kamping::recv_buf() is given.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described
/// above.
/// @return If no message is available return a \c nullopt, else return a \c std::optional wrapping an \ref
/// kamping::MPIResult
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::try_recv(Args... args) const {
    // Check parameters
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(),
        KAMPING_OPTIONAL_PARAMETERS(recv_buf, tag, source, status)
    );

    //  Get the recv buffer
    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<recv_value_type_tparam>>));
    auto&& recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        );
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    static_assert(
        !std::is_same_v<recv_value_type, internal::unused_tparam>,
        "No recv_buf parameter provided and no receive value given as template parameter. One of these is required."
    );

    // Get the source parameter. If the parameter is not given, use MPI_ANY_SOURCE.
    using default_source_buf_type = decltype(kamping::source(rank::any));
    auto&& source_param =
        internal::select_parameter_type_or_default<internal::ParameterType::source, default_source_buf_type>(
            {},
            args...
        );
    KASSERT(internal::is_valid_rank_in_comm(source_param, *this, true, true));
    int source = source_param.rank_signed();

    // Get the tag parameter. If the parameter is not given, use MPI_ANY_TAG.
    using default_tag_buf_type = decltype(kamping::tag(tags::any));
    auto&& tag_param =
        internal::select_parameter_type_or_default<internal::ParameterType::tag, default_tag_buf_type>({}, args...);
    constexpr auto tag_type = std::remove_reference_t<decltype(tag_param)>::tag_type;
    if constexpr (tag_type == internal::TagType::value) {
        int tag = tag_param.tag();
        KASSERT(
            Environment<>::is_valid_tag(tag),
            "invalid tag " << tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
        );
    }
    int tag = tag_param.tag();

    // Get the status parameter. If the parameter is not given, return the status (we have to create it anyhow).
    using default_status_param_type = decltype(kamping::status_out());
    auto&& status_param =
        internal::select_parameter_type_or_default<internal::ParameterType::status, default_status_param_type>(
            {},
            args...
        );

    // Use a matched probe to check if a message with the given source and tag is available for receiving.
    int         msg_avail;
    MPI_Message message;

    Status               status;
    [[maybe_unused]] int err = MPI_Improbe(source, tag, _comm, &msg_avail, &message, &status.native());
    THROW_IF_MPI_ERROR(err, MPI_Improbe);

    // If a message is available, receive it using a matched receive
    if (msg_avail) {
        int const count = asserting_cast<int>(status.template count<recv_value_type>());
        KASSERT(count >= 0, "Received a message with a negative number of elements (count).", assert::light);
        KASSERT(source == MPI_ANY_SOURCE || source == status.source_signed(), "source mismatch", assert::light);
        KASSERT(tag == MPI_ANY_TAG || tag == status.tag(), "tag mismatch", assert::light);
        // Update source and tag to the tag and source of the received message. These might be different from the
        // requested tag and source in case MPI_ANY_{SOURCE,TAG} was requested.
        source = status.source_signed();
        tag    = status.tag();

        // Ensure that we do not touch the recv buffer if MPI_PROC_NULL is passed, because this is what the standard
        // guarantees.
        if constexpr (std::remove_reference_t<decltype(source_param)>::rank_type != internal::RankType::null) {
            auto compute_required_recv_buf_size = [&] {
                return asserting_cast<size_t>(count);
            };
            recv_buf.resize_if_requested(compute_required_recv_buf_size);
        }

        // Use a matched receive to receive exactly the message we probed. This ensures this method is thread-safe.
        err = MPI_Mrecv(
            recv_buf.data(),                                   // buf
            count,                                             // count
            mpi_datatype<recv_value_type>(),                   // datatype
            &message,                                          // message
            internal::status_param_to_native_ptr(status_param) // status
        );
        THROW_IF_MPI_ERROR(err, MPI_Recv);

        // Build the result object and return.
        return std::optional{make_mpi_result(std::move(recv_buf), std::move(status_param))};
    } else {
        // There was to mesage to receive, thus return std::nullopt.
        return std::optional<decltype(make_mpi_result(std::move(recv_buf), std::move(status_param)))>{};
    }
}
