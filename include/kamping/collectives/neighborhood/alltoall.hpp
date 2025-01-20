// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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

#include <cstddef>
#include <tuple>
#include <type_traits>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/collectives_helpers.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"
#include "kamping/topology_communicator.hpp"

/// @addtogroup kamping_collectives
/// @{

/// @brief Wrapper for \c MPI_Neighbor_alltoall.
///
/// This wrapper for \c MPI_Neighbor_alltoall sends the same amount of data
/// from a rank i to each of its neighbour j for which an edge (i,j) in the communication graph exists. The following
/// buffers are required:
/// @todo check again once the concrete semantics (potential differing number of send/recv counts) of
/// MPI_Neighbor_alltoall has been clarified.
/// - \ref kamping::send_buf() containing the data that is sent to each neighbor. This buffer has to be divisible by the
/// out degree unless a send_count or a send_type is explicitly given as parameter.
///
/// The following parameters are optional:
/// - \ref kamping::send_count() specifying how many elements are sent. If
/// omitted, the size of send buffer divided by number of outgoing neighbors is used.
/// This has to be the same on all ranks.
/// This parameter is mandatory if \ref kamping::send_type() is given.
///
/// - \ref kamping::recv_count() specifying how many elements are received. If
/// omitted, the value of send_counts will be used.
/// This parameter is mandatory if \ref kamping::recv_type() is given.
///
/// - \ref kamping::recv_buf() specifying a buffer for the output. A buffer of at least
/// `recv_count * in degree` is required.
///
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype is
/// derived automatically based on send_buf's underlying \c value_type.
///
/// - \ref kamping::recv_type() specifying the \c MPI datatype to use as recv type. If omitted, the \c MPI datatype is
/// derived automatically based on recv_buf's underlying \c value_type.
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional buffers described above.
/// @return Result type wrapping the output parameters to be returned by value.
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
auto kamping::TopologyCommunicator<DefaultContainerType, Plugins...>::neighbor_alltoall(Args... args) const {
    using namespace internal;
    KAMPING_CHECK_PARAMETERS(
        Args,
        KAMPING_REQUIRED_PARAMETERS(send_buf),
        KAMPING_OPTIONAL_PARAMETERS(recv_buf, send_count, recv_count, send_type, recv_type)
    );
    // Get the buffers
    auto const send_buf =
        internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
    using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
    using default_recv_value_type = std::remove_const_t<send_value_type>;

    using default_recv_buf_type = decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
    auto recv_buf =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
            std::tuple(),
            args...
        )
            .template construct_buffer_or_rebind<DefaultContainerType>();
    using recv_value_type = typename std::remove_reference_t<decltype(recv_buf)>::value_type;

    static_assert(!std::is_const_v<recv_value_type>, "The receive buffer must not have a const value_type.");

    auto [send_type, recv_type] =
        internal::determine_mpi_datatypes<send_value_type, recv_value_type, decltype(recv_buf)>(args...);
    [[maybe_unused]] constexpr bool recv_type_has_to_be_deduced = has_to_be_computed<decltype(recv_type)>;

    // Get the send counts
    using default_send_count_type = decltype(kamping::send_count_out());
    auto send_count =
        internal::select_parameter_type_or_default<internal::ParameterType::send_count, default_send_count_type>(
            std::tuple(),
            args...
        )
            .construct_buffer_or_rebind();
    constexpr bool do_compute_send_count = internal::has_to_be_computed<decltype(send_count)>;
    if constexpr (do_compute_send_count) {
        send_count.underlying() =
            this->out_degree() == 0 ? 0 : asserting_cast<int>(send_buf.size() / this->out_degree());
    }
    // Get the recv counts
    using default_recv_count_type = decltype(kamping::recv_count_out());
    auto recv_count =
        internal::select_parameter_type_or_default<internal::ParameterType::recv_count, default_recv_count_type>(
            std::tuple(),
            args...
        )
            .construct_buffer_or_rebind();

    constexpr bool do_compute_recv_count = internal::has_to_be_computed<decltype(recv_count)>;
    if constexpr (do_compute_recv_count) {
        recv_count.underlying() = send_count.get_single_element();
    }

    KASSERT(
        // @todo check this condition once we know the exact intended semantics of neighbor_alltoall
        (!do_compute_send_count || this->out_degree() == 0 || send_buf.size() % this->out_degree() == 0lu),
        "There are no send counts given and the number of elements in send_buf is not divisible by the number "
        "of "
        "(out) neighbors.",
        assert::light
    );

    auto compute_required_recv_buf_size = [&]() {
        return asserting_cast<size_t>(recv_count.get_single_element()) * this->in_degree();
    };
    recv_buf.resize_if_requested(compute_required_recv_buf_size);
    KASSERT(
        // if the recv type is user provided, kamping cannot make any assumptions about the required size of the
        // recv buffer
        !recv_type_has_to_be_deduced || recv_buf.size() >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // These KASSERTs are required to avoid a false warning from g++ in release mode
    KASSERT(send_buf.data() != nullptr, assert::light);
    KASSERT(recv_buf.data() != nullptr, assert::light);

    [[maybe_unused]] int err = MPI_Neighbor_alltoall(
        send_buf.data(),                 // send_buf
        send_count.get_single_element(), // send_count
        send_type.get_single_element(),  // send_type
        recv_buf.data(),                 // recv_buf
        recv_count.get_single_element(), // recv_count
        recv_type.get_single_element(),  // recv_type
        this->mpi_communicator()         // comm
    );

    this->mpi_error_hook(err, "MPI_Alltoall");
    return make_mpi_result<std::tuple<Args...>>(
        std::move(recv_buf),   // recv_buf
        std::move(send_count), // send_count
        std::move(recv_count), // recv_count
        std::move(send_type),  // send_type
        std::move(recv_type)   // recv_type
    );
}
/// @}
