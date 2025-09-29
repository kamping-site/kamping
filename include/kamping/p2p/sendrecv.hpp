// This file is part of KaMPIng.
//
// Copyright 2022-2025 The KaMPIng Authors
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

#include "kamping/comm_helper/infer_rbuf_vals_from.hpp"
#include "kamping/communicator.hpp"

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename SBuff, typename RBuff, typename StatusObject>
requires kamping::DataBufferConcept<SBuff> && kamping::DataBufferConcept<RBuff> && kamping::SendDataBuffer<
    SBuff> && kamping::RecvDataBuffer<RBuff>
auto kamping::Communicator<DefaultContainerType, Plugins...>::sendrecv(
    SBuff&& sbuf, RBuff&& rbuf, int dest, int send_tag, int source, int recv_tag, StatusObject status_param
) const {
    if (send_tag == MPI_ANY_TAG) {
        send_tag = default_tag();
    }

    infer<CommType::sendrecv>(sbuf, rbuf, source, dest, *this);

    int send_size = asserting_cast<int>(std::ranges::size(sbuf));
    int recv_size = asserting_cast<int>(std::ranges::size(rbuf));

    KASSERT(
        Environment<>::is_valid_tag(send_tag),
        "invalid send tag " << send_tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
    );

    KASSERT(recv_size >= send_size, "The receive buffer is not large enough", assert::light);

    auto status = status_param.construct_buffer_or_rebind();

    [[maybe_unused]] int err = MPI_Sendrecv(
        std::ranges::data(sbuf),                     // send_buff
        send_size,                                   // send_count
        type(sbuf),                                  // send_data_type
        dest,                                        // destination
        send_tag,                                    // send_tag
        std::ranges::data(rbuf),                     // recv_buff
        recv_size,                                   // recv_count
        type(rbuf),                                  // recv_data_type
        source,                                      // source
        recv_tag,                                    // recv_tag
        this->mpi_communicator(),                    // comm
        internal::status_param_to_native_ptr(status) // status
    );
    this->mpi_error_hook(err, "MPI_Sendrecv");

    return std::pair<SBuff, RBuff>(std::forward<SBuff>(sbuf), std::forward<RBuff>(rbuf));
}
/// @}