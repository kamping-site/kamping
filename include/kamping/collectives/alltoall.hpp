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

#include <tuple>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/comm_helper/infer_rbuf_vals_from.hpp"
#include "kamping/communicator.hpp"

/// @addtogroup kamping_collectives
/// @{

/// @brief Wrapper for \c MPI_Alltoall.
///
/// This wrapper for \c MPI_Alltoall sends the same amount of data from each rank to each rank. The following
/// buffers are required:
/// - Send buffer according to the \ref SendDataBuffer concept. The size returned by the send buffer has to be
/// dividable by the number of PEs on the current communicator. The amount of data sent by each PE is the size of the
/// send buffer divided by the number of PEs.
///
/// - Receive buffer according to the \ref RecvDataBuffer concept.

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <kamping::SendDataBuffer SBuff, kamping::RecvDataBuffer RBuff>
auto kamping::Communicator<DefaultContainerType, Plugins...>::alltoall(SBuff&& sbuf, RBuff&& rbuf) const {
    infer<CommType::alltoall>(sbuf, rbuf, *this);

    size_t send_size = std::ranges::size(sbuf);
    size_t recv_size = std::ranges::size(rbuf);

    KASSERT(
        (std::ranges::size(sbuf) % size() == 0lu),
        "The number of elements in the send buffer is not divisible by the number of ranks in the communicator.",
        assert::light
    );

    KASSERT(recv_size >= send_size, "The receive buffer is not large enough", assert::light);

    [[maybe_unused]] int err = MPI_Alltoall(
        std::ranges::data(sbuf),                 // send_buf
        asserting_cast<int>(send_size / size()), // send_count
        type(sbuf),                              // send_type
        std::ranges::data(rbuf),                 // recv_buf
        asserting_cast<int>(recv_size / size()), // recv_count
        type(rbuf),                              // recv_type
        this->mpi_communicator()                 // comm
    );

    this->mpi_error_hook(err, "MPI_Alltoall");
    return std::tuple<SBuff, RBuff>(std::forward<SBuff>(sbuf), std::forward<RBuff>(rbuf));
}

/// @brief Wrapper for \c MPI_Alltoallv.
///
/// This wrapper for \c MPI_Alltoallv sends the different amounts of data from each rank to each rank. The following
/// buffers are required:
/// - Send buffer has to follow the \ref SendDataBuffer concept and the \ref ExtendedDataBuffer concept.
/// The following functions of the recv buffer are optional:
///     * set_displacements(...) the send displacements will be computed as the exclusive prefix-sum of the send counts.

/// - Receive buffer according to the \ref RecvDataBuffer concept and the \ref ExtendedDataBuffer concept.
/// The following functions of the recv buffer are optional:
///     * set_size_v(...), the recv counts will be computed using additional communication.
///     * set_displacements(...) the recv displacements will be computed as the exclusive prefix-sum of the recv counts.

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <kamping::SendDataBuffer SBuff, kamping::RecvDataBuffer RBuff>
requires kamping::ExtendedDataBuffer<SBuff> && kamping::ExtendedDataBuffer<RBuff>
auto kamping::Communicator<DefaultContainerType, Plugins...>::alltoallv(SBuff&& sbuf, RBuff&& rbuf) const {
    auto& send_counts = sbuf.size_v();
    // Assert the send counts first, because they may be used to infer the recv counts
    KASSERT(std::ranges::size(send_counts) >= this->size(), "Send counts buffer is not large enough.", assert::light);

    infer<CommType::alltoallv>(sbuf, rbuf, *this);

    auto& send_displs = sbuf.displs();
    KASSERT(std::ranges::size(send_displs) >= this->size(), "Send displs buffer is not large enough.", assert::light);

    auto& recv_counts = rbuf.size_v();
    KASSERT(std::ranges::size(recv_counts) >= this->size(), "Recv counts buffer is not large enough.", assert::light);

    auto& recv_displs = rbuf.displs();
    KASSERT(std::ranges::size(recv_displs) >= this->size(), "Recv displs buffer is not large enough.", assert::light);

    auto compute_recv_size = [&]() {
        int recv_buf_size = 0;
        auto counts_ptr = std::ranges::data(recv_counts);
        auto displs_ptr = std::ranges::data(recv_displs);
        for (size_t i = 0; i < this->size(); ++i) {
            recv_buf_size = std::max(recv_buf_size, *(counts_ptr + i) + *(displs_ptr + i));
        }
        return kamping::asserting_cast<size_t>(recv_buf_size);
    };

    KASSERT(std::ranges::size(rbuf) >= compute_recv_size(), "Recv buffer is not large enough.", assert::light);


    // Do the actual alltoallv
    [[maybe_unused]] int err = MPI_Alltoallv(
        std::ranges::data(sbuf),        // send_buf
        std::ranges::data(send_counts), // send_counts
        std::ranges::data(send_displs), // send_displs
        type(sbuf),                     // send_type
        std::ranges::data(rbuf),        // recv_buf
        std::ranges::data(recv_counts), // recv_counts
        std::ranges::data(recv_displs), // recv_displs
        type(rbuf),                     // recv_type
        this->mpi_communicator()        // comm
    );

    this->mpi_error_hook(err, "MPI_Alltoallv");

    return std::pair<SBuff, RBuff>(std::forward<SBuff>(sbuf), std::forward<RBuff>(rbuf));
}
/// @}
