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

#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/implementation_helpers.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

///// @addtogroup kamping_p2p
/// @{

// @brief Wrapper for \c MPI_Probe.
///
/// This wraps \c MPI_Probe. This operation probes for a message and returns a
/// status object with info about the probe message. The following parameters
/// are optional:
/// - \ref kamping::source() probe for messages sent from this source rank.
/// Defaults to probing for an arbitrary source, i.e. \c source(rank::any).
/// - \ref kamping::tag() probe for messages with this tag. Defaults to probing
/// for an arbitrary tag, i.e. \c tag(tags::any).
/// - \c kamping::status(ignore<>) or \ref kamping::status_out(). Returns info about
/// the probed message by setting the appropriate fields in the status object
/// passed by the user. If \ref kamping::status_out() is passed, constructs a
/// status object which may be retrieved by the user.
/// The status can be ignored by passing \c kamping::status(kamping::ignore<>). This is the default.
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional buffers described
/// above.
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename... Args>
auto kamping::Communicator<DefaultContainerType, Plugins...>::probe(Args... args) const {
    KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(tag, source, status));

    using default_source_buf_type = decltype(kamping::source(rank::any));

    auto&& source =
        internal::select_parameter_type_or_default<internal::ParameterType::source, default_source_buf_type>(
            {},
            args...
        );

    using default_tag_buf_type = decltype(kamping::tag(tags::any));

    auto&& tag_param =
        internal::select_parameter_type_or_default<internal::ParameterType::tag, default_tag_buf_type>({}, args...);
    int tag = tag_param.tag();

    constexpr auto tag_type = std::remove_reference_t<decltype(tag_param)>::tag_type;
    if constexpr (tag_type == internal::TagType::value) {
        KASSERT(
            Environment<>::is_valid_tag(tag),
            "invalid tag " << tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
        );
    }

    using default_status_param_type = decltype(kamping::status(kamping::ignore<>));

    auto status =
        internal::select_parameter_type_or_default<internal::ParameterType::status, default_status_param_type>(
            {},
            args...
        )
            .construct_buffer_or_rebind();

    KASSERT(internal::is_valid_rank_in_comm(source, *this, true, true), "Invalid source rank.");

    [[maybe_unused]] int err = MPI_Probe(
        source.rank_signed(),                        // source
        tag,                                         // tag
        this->mpi_communicator(),                    // comm
        internal::status_param_to_native_ptr(status) // status
    );
    this->mpi_error_hook(err, "MPI_Probe");

    return internal::make_mpi_result<std::tuple<Args...>>(std::move(status));
}
/// @}
