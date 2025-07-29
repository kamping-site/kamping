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
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/result.hpp"

/// @addtogroup kamping_collectives
/// @{

/// @brief Wrapper for \c MPI_Allgather.
///
/// This wrapper for \c MPI_Allgather collects the same amount of data from each rank to all ranks. It is semantically
/// equivalent to performing a \c gather() followed by a broadcast of the collected data.
///
/// The following parameters are required:
/// - \ref kamping::send_buf() containing the data that is sent to the root. This buffer has to be the same size at
/// each rank. See allgather_v if the amounts differ.
///
/// The following parameters are optional:
/// - \ref kamping::send_count() specifying how many elements are sent. If
/// omitted, the size of the send buffer is used. This parameter is mandatory if \ref kamping::send_type() is given.
///
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype is
/// derived automatically based on send_buf's underlying \c value_type.
///
/// - \ref kamping::recv_count() specifying how many elements are received. If
/// omitted, the value of send_counts will be used. This parameter is mandatory if \ref kamping::recv_type() is given.
///
/// - \ref kamping::recv_buf() specifying a buffer for the output. Afterwards, this buffer will contain
/// all data from all send buffers. This requires a size of the buffer of at least `recv_counts * communicator size`.
///
/// - \ref kamping::recv_type() specifying the \c MPI datatype to use as recv type. If omitted, the \c MPI datatype is
/// derived automatically based on recv_buf's underlying \c value_type.
///
/// In-place allgather is supported by passing send_recv_buf() as parameter. This changes the requirements for the other
/// parameters, see \ref Communicator::allgather_inplace.
///
/// @tparam Args Automatically deduced template parameters.
/// @param args All required and any number of the optional parameters described above.
/// @return Result object wrapping the output parameters to be returned by value.
///
/// @see \ref docs/parameter_handling.md for general information about parameter handling in KaMPIng.
/// <hr>
/// \include{doc} docs/resize_policy.dox
template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
template <typename SBuff, typename RBuff>
//requires DataBuffer(SBuff) && DataBuffer(RBuff) && SendDataBuffer(SBuff) && RecvDataBuffer(RBuff)
auto kamping::Communicator<DefaultContainerType, Plugins...>::allgather(SBuff&& sbuf, RBuff&& rbuf) const {
    using namespace kamping::internal;

    using send_type = typename std::decay_t<SBuff>::value_type;
    using recv_type = typename std::decay_t<RBuff>::value_type;

    auto send_count = sbuf.size();

    KASSERT(
        is_same_on_all_ranks(send_count),
        "All PEs have to send the same number of elements. Use allgatherv, if you want to send a different number "
        "of "
        "elements.",
        assert::light_communication
    );



    auto compute_required_recv_buf_size = [&]() {
        return asserting_cast<size_t>(send_count * size());
    };

    //rbuf.resize_if_requested(compute_required_recv_buf_size);

    auto recv_count = rbuf.size();

    KASSERT(
        recv_count >= compute_required_recv_buf_size(),
        "Recv buffer is not large enough to hold all received elements.",
        assert::light
    );

    // error code can be unused if KTHROW is removed at compile time
    [[maybe_unused]] int err = MPI_Allgather(
        sbuf.data(),
        asserting_cast<int>(send_count),
        mpi_datatype<send_type>(),
        rbuf.data(),
        asserting_cast<int>(send_count),
        mpi_datatype<recv_type>(),
        this->mpi_communicator()
    );
    this->mpi_error_hook(err, "MPI_Allgather");

    return std::pair<SBuff, RBuff>(std::forward<SBuff>(sbuf), std::forward<RBuff>(rbuf));
}
