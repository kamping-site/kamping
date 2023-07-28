// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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

/// @brief Wrapper for \c MPI_Recv.
///
/// This wraps \c MPI_Recv. This operation performs a standard blocking receive.
/// If the \ref kamping::send_counts() parameter is not specified, this first performs a probe, followed by a receive of
/// the probed message with the probed message size.
///
/// The following parameters are optional:
/// - \ref kamping::recv_buf() the buffer to receive the message into. If possible, this buffer will be resized to
/// accommodate the number of elements to receive. Use \c kamping::Span with enough space if you do not want the buffer
/// to be resized. If no \ref kamping::recv_buf() is provided, the type that should be received has to be passed as a
/// template parameter to \c recv().
/// - \ref kamping::tag() recv message with this tag. Defaults to receiving
/// for an arbitrary tag, i.e. \c tag(tags::any).
/// - \ref kamping::source() receive a message sent from this source rank.
/// Defaults to probing for an arbitrary source, i.e. \c source(rank::any).
/// - \ref kamping::status() or \ref kamping::status_out(). Returns info about
/// the received message by setting the appropriate fields in the status object
/// passed by the user. If \ref kamping::status_out() is passed, constructs a
/// status object which may be retrieved by the user. The status can be ignored by
/// passing \c kamping::status(kamping::ignore<>). This is the default.
///
/// The following parameter is optional, but leads to an additional call to \c MPI_Probe if not present:
/// - \ref kamping::send_counts() the number of elements to receive. Will be probed before receiving if not given.
///
/// @tparam recv_value_type_tparam The type that is received. Only required when no \ref kamping::recv_buf() is given.
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional buffers described
/// above.
template <template <typename...> typename DefaultContainerType, template <typename> typename... Plugins>
template <typename recv_value_type_tparam /* = kamping::internal::unused_tparam */, typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::recv(Args... args) const {
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(),
        KAMPING_OPTIONAL_PARAMETERS(recv_buf, tag, source, recv_counts, status)
    );
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

    using default_source_buf_type = decltype(kamping::source(rank::any));

    auto&& source_param =
        internal::select_parameter_type_or_default<internal::ParameterType::source, default_source_buf_type>(
            {},
            args...
        );

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

    using default_status_param_type = decltype(kamping::status(kamping::ignore<>));

    auto&& status =
        internal::select_parameter_type_or_default<internal::ParameterType::status, default_status_param_type>(
            {},
            args...
        );

    // Get the optional recv_count parameter. If the parameter is not given,
    // allocate a new container.
    using default_recv_count_type = decltype(kamping::recv_counts_out(alloc_new<int>));
    auto&& recv_count_param =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_count_type>(
            std::tuple(),
            args...
        );

    KASSERT(internal::is_valid_rank_in_comm(source_param, *this, true, true));
    int            source                         = source_param.rank_signed();
    int            tag                            = tag_param.tag();
    constexpr bool recv_count_is_output_parameter = internal::has_to_be_computed<decltype(recv_count_param)>;
    static_assert(
        std::remove_reference_t<decltype(recv_count_param)>::is_single_element,
        "recv_counts() parameter must be a single value."
    );
    if constexpr (recv_count_is_output_parameter) {
        Status probe_status      = this->probe(source_param.clone(), tag_param.clone(), status_out()).extract_status();
        source                   = probe_status.source_signed();
        tag                      = probe_status.tag();
        *recv_count_param.data() = asserting_cast<int>(probe_status.template count<recv_value_type>());
    }

    // Ensure that we do not touch the recv buffer if MPI_PROC_NULL is passed,
    // because this is what the standard guarantees.
    if constexpr (std::remove_reference_t<decltype(source_param)>::rank_type != internal::RankType::null) {
        recv_buf.resize(asserting_cast<size_t>(recv_count_param.get_single_element()));
    }

    [[maybe_unused]] int err = MPI_Recv(
        recv_buf.data(),                                            // buf
        asserting_cast<int>(recv_count_param.get_single_element()), // count
        mpi_datatype<recv_value_type>(),                            // dataype
        source,                                                     // source
        tag,                                                        // tag
        this->mpi_communicator(),                                   // comm
        status.native_ptr()                                         // status
    );
    THROW_IF_MPI_ERROR(err, MPI_Recv);

    return make_mpi_result(std::move(recv_buf), std::move(recv_count_param), std::move(status));
}
