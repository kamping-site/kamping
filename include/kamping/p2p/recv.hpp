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

#include <ranges>
#include <type_traits>
#include <utility>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/comm_helper/infer_rbuf_vals_from.hpp"
#include "kamping/communicator.hpp"
#include "kamping/implementation_helpers.hpp"

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename RBuff, typename StatusObject>
requires kamping::DataBufferConcept<RBuff> && kamping::RecvDataBuffer<RBuff>
auto kamping::Communicator<DefaultContainerType, Plugins...>::recv(
    RBuff&& rbuf, int source, int tag, StatusObject status_param
) const {
    infer<CommType::recv>(rbuf, *this);

    static_assert(
        StatusObject::parameter_type == internal::ParameterType::status,
        "Only status parameters are allowed."
    );
    auto status = status_param.construct_buffer_or_rebind();

    [[maybe_unused]] int err = MPI_Recv(
        std::ranges::data(rbuf),                      // buf
        asserting_cast<int>(std::ranges::size(rbuf)), // count
        type(rbuf),                                   // datatype
        source,                                       // source
        tag,                                          // tag
        this->mpi_communicator(),                     // comm
        internal::status_param_to_native_ptr(status)  // status
    );
    this->mpi_error_hook(err, "MPI_Recv");

    return std::forward<RBuff>(rbuf);
}
