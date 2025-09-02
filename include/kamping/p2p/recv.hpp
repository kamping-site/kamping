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

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/comm_helper/infer_rbuf_vals_from.hpp"
#include "kamping/communicator.hpp"
#include "kamping/implementation_helpers.hpp"

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename RBuff>
requires kamping::DataBufferConcept<RBuff> && kamping::RecvDataBuffer<RBuff>
// FIXME source, tag, status should not be part of the data buffer
// handle status similar to Request::wait (request.hpp)
auto kamping::Communicator<DefaultContainerType, Plugins...>::recv(RBuff&& rbuf/* , int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG, Status ... */) const {
    using namespace kamping::internal;

    infer<CommType::recv>(rbuf, *this);
    using recv_type = std::ranges::range_value_t<RBuff>;

    int source = MPI_ANY_SOURCE;
    if constexpr (HasSource<RBuff>) {
        source = rbuf.source();
    }

    int tag = MPI_ANY_TAG;
    if constexpr (HasTag<RBuff>) {
        tag = rbuf.tag();
    }

    auto status = MPI_Status{};
    if constexpr (HasStatus<RBuff>) {
        status = rbuf.status();
    }

    auto t = rbuf.size();

    [[maybe_unused]] int err = MPI_Recv(
        rbuf.data(),                          // buf
        asserting_cast<int>(std::size(rbuf)), // count
        mpi_datatype<recv_type>(),            // datatype
        source,                               // source
        tag,                                  // tag
        this->mpi_communicator(),             // comm
        &status                               // status
    );
    this->mpi_error_hook(err, "MPI_Recv");

    return std::forward<RBuff>(rbuf);
}
