// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
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
#include "kamping/implementation_helpers.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/helpers.hpp"
#include "kamping/p2p/probe.hpp"
#include "kamping/parameter_objects.hpp"
#include "kamping/result.hpp"
#include "kamping/status.hpp"

///// @addtogroup kamping_p2p
/// @{

// @brief Wrapper for \c MPI_Sendrecv.
///
/// This wraps \c MPI_Sendrecv. This operation performs a blocking send and receive operation. If the
/// \ref kamping::recv_counts() parameter is not specified, this first performs a probe, followed by a receive of
/// the probed message with the probed message size.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent.
///
///
/// - \ref kamping::destination() the receiving rank.
///
/// The following parameter is optional, but leads to an additional call to \c MPI_Probe if not present:
/// - \ref kamping::recv_count() the number of elements to receive. Will be calculated using an additional sendrecv
/// if not given.
///
/// The following parameters are optional:
/// - \ref kamping::recv_buf() the buffer to receive the message into.  The buffer's underlying
/// storage must be large enough to hold all received elements. If no \ref kamping::recv_buf() is provided, the \c
/// value_type of the recv buffer has to be passed as a template parameter to \c sendrecv().
///
/// - \ref kamping::send_count() specifying how many elements of the buffer are sent. If omitted, the size of the send
/// buffer is used as a default.
///
/// - \ref kamping::source() receive a message sent from this source rank. Defaults to probing for an arbitrary source,
/// i.e. \c source(rank::any).
///
/// - \ref kamping::send_tag() send message with this tag. Defaults to sending for an arbitrary tag, i.e. \c
/// tag(tags::any).
///
/// - \ref kamping::recv_tag() receive message with this tag. Defaults to receiving for an arbitrary tag, i.e. \c
/// tag(tags::any).
///
/// - \c kamping::status(ignore<>) or \ref kamping::status_out(). Returns info about the received message by setting the
/// appropriate fields in the status object passed by the user. If \ref kamping::status_out() is passed, constructs a
/// status object which may be retrieved by the user. The status can be ignored by passing \c
/// kamping::status(kamping::ignore<>). This is the default.
///
/// @tparam Args Automatically deduced template parameters.
/// @tparam recv_value_type_tparam The type of the message to be received.
/// @param args All required and any number of the optional parameters described above.
/// @return Result object wrapping the output parameters to be returned by value.
///
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
/// <hr>

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::sendrecv(Args... args) const {
    using namespace kamping::internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf, destination),
        KAMPING_OPTIONAL_PARAMETERS(
            send_count,
            send_type,
            send_tag,
            recv_buf,
            recv_tag,
            source,
            recv_type,
            status,
            recv_count
        )
    );

    auto send_buf = internal::select_parameter_type<internal::ParameterType::send_buf>(args...)
                        .template construct_buffer_or_rebind<UnusedRebindContainer, serialization_support_tag>();

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

    using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    auto send_type        = internal::determine_mpi_send_datatype<send_value_type>(args...);

    auto const&    destination = internal::select_parameter_type<internal::ParameterType::destination>(args...);
    constexpr auto rank_type   = std::remove_reference_t<decltype(destination)>::rank_type;
    static_assert(
        rank_type == RankType::value || rank_type == RankType::null,
        "Please provide an explicit destination or destination(ranks::null)."
    );

    using default_send_tag_type = decltype(kamping::tag(this->default_tag()));
    auto&& send_tag_param =
        internal::select_parameter_type_or_default<internal::ParameterType::send_tag, default_send_tag_type>(
            std::tuple(this->default_tag()),
            args...
        );

    // this ensures that the user does not try to pass MPI_ANY_TAG, which is not allowed for the send tag
    static_assert(
        std::remove_reference_t<decltype(send_tag_param)>::tag_type == TagType::value,
        "Please provide a send tag for the message."
    );
    int send_tag = send_tag_param.tag();
    KASSERT(
        Environment<>::is_valid_tag(send_tag),
        "invalid send tag " << send_tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
    );

    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<recv_value_type_tparam>>));
    auto recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType, internal::serialization_support_tag>();

    KASSERT(
        !(send_buf.size() == recv_buf.size() && send_buf.data() == recv_buf.data()),
        "Send buffer and recv buffer can not be the same."
    );

    using default_recv_count_type = decltype(kamping::recv_count_out());
    auto recv_count_param =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_count, default_recv_count_type>(
            std::tuple(),
            args...
        )
            .construct_buffer_or_rebind();

    constexpr bool is_recv_serialization_used = internal::buffer_uses_serialization<decltype(recv_buf)>;

    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
    static_assert(
        !std::is_same_v<recv_value_type, internal::unused_tparam>,
        "No recv_buf parameter provided and no receive value given as template parameter. One of these is required."
    );
    auto recv_type = internal::determine_mpi_recv_datatype<recv_value_type, decltype(recv_buf)>(args...);
    [[maybe_unused]] constexpr bool recv_type_is_in_param = !internal::has_to_be_computed<decltype(recv_type)>;

    using default_recv_tag_type = decltype(kamping::tag(tags::any));
    auto&& recv_tag_param =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_tag, default_recv_tag_type>(
            {},
            args...
        );
    constexpr auto tag_type = std::remove_reference_t<decltype(recv_tag_param)>::tag_type;
    if constexpr (tag_type == internal::TagType::value) {
        int recv_tag = recv_tag_param.tag();
        KASSERT(
            Environment<>::is_valid_tag(recv_tag),
            "invalid recv tag " << recv_tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
        );
    }

    using default_status_param_type = decltype(kamping::status(kamping::ignore<>));
    auto status =
        internal::select_parameter_type_or_default<internal::ParameterType::status, default_status_param_type>(
            {},
            args...
        )
            .construct_buffer_or_rebind();

    using default_source_buf_type = decltype(kamping::source(rank::any));
    auto&& source_param =
        internal::select_parameter_type_or_default<internal::ParameterType::source, default_source_buf_type>(
            {},
            args...
        );

    KASSERT(internal::is_valid_rank_in_comm(source_param, *this, true, true));
    int source = source_param.rank_signed();

    // Calculate rev_count if not given by calling sendrecv with the send_count
    if constexpr (internal::has_to_be_computed<decltype(recv_count_param)>) {
        this->sendrecv(
            kamping::destination(destination.rank_signed()),
            kamping::send_count(1),
            kamping::send_buf(send_count.get_single_element()),
            kamping::recv_count(1),
            kamping::recv_buf(recv_count_param)
        );
    }

    // Ensure that we do not touch the recv buffer if MPI_PROC_NULL is passed,
    // because this is what the standard guarantees.
    // Resize the buffer if requested and ensure that it is large enough
    if constexpr (std::remove_reference_t<decltype(source_param)>::rank_type != internal::RankType::null) {
        auto compute_required_recv_buf_size = [&] {
            return asserting_cast<size_t>(recv_count_param.get_single_element());
        };
        recv_buf.resize_if_requested(compute_required_recv_buf_size);
        KASSERT(
            // if the recv type is user provided, kamping cannot make any assumptions about the required size of the
            // recv buffer
            recv_type_is_in_param || recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );
    }

    [[maybe_unused]] int err = MPI_Sendrecv(
        send_buf.data(),                             // send_buff
        send_count.get_single_element(),             // send_count
        send_type.get_single_element(),              // send_data_type
        destination.rank_signed(),                   // destination
        send_tag,                                    // send_tag
        recv_buf.data(),                             // recv_buff
        recv_count_param.get_single_element(),       // recv_count
        recv_type.get_single_element(),              // recv_data_type
        source,                                      // source
        recv_tag_param.tag(),                        // recv_tag
        this->mpi_communicator(),                    // comm
        internal::status_param_to_native_ptr(status) // status
    );
    this->mpi_error_hook(err, "MPI_Sendrecv");

    return internal::make_mpi_result<std::tuple<Args...>>(
        deserialization_repack<is_recv_serialization_used>(std::move(recv_buf)),
        std::move(recv_count_param),
        std::move(status),
        std::move(recv_type)
    );
}
/// @}