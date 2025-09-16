// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
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

#include <numeric>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/collectives_helpers.hpp"
#include "kamping/comm_helper/infer_rbuf_vals_from.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_datatype.hpp"

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename SBuff, typename RBuff>
requires kamping::DataBufferConcept<SBuff> && kamping::DataBufferConcept<RBuff> && kamping::SendDataBuffer<
    SBuff> && kamping::RecvDataBuffer<RBuff>
auto kamping::Communicator<DefaultContainerType, Plugins...>::allgather(SBuff&& sbuf, RBuff&& rbuf) const {
    using namespace kamping::internal;

    infer<CommType::allgather>(sbuf, rbuf, *this);

    using send_type = std::ranges::range_value_t<SBuff>;
    using recv_type = std::ranges::range_value_t<RBuff>;

    auto   send_size = std::size(sbuf);
    size_t recv_size = std::size(rbuf);

    KASSERT(
        is_same_on_all_ranks(send_size),
        "All PEs have to send the same number of elements. Use allgatherv, if you want to send a different number "
        "of "
        "elements.",
        assert::light_communication
    );

    KASSERT(recv_size >= send_size, "The receive buffer is not large enough", assert::light);

    // error code can be unused if KTHROW is removed at compile time
    [[maybe_unused]] int err = MPI_Allgather(
        sbuf.data(),
        asserting_cast<int>(send_size),
        mpi_datatype<send_type>(),
        rbuf.data(),
        asserting_cast<int>(recv_size / size()),
        mpi_datatype<recv_type>(),
        this->mpi_communicator()
    );
    this->mpi_error_hook(err, "MPI_Allgather");

    return std::pair<SBuff, RBuff>(std::forward<SBuff>(sbuf), std::forward<RBuff>(rbuf));
}
