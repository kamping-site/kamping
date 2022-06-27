// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <numeric>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/parameter_factories.hpp"

using namespace ::kamping;
using namespace ::testing;

int main() {
    using namespace kamping;
    kamping::Communicator comm;

    /// @todo Expand these examples, once we have send_recv_buf as unnamed first parameter.
    /// @todo Expand these examples, once we have bcast_single.

    // You can broadcast a single element from the communicators root rank to all other ranks using:
    size_t value = comm.rank();
    comm.bcast(send_recv_buf(value)); // Two MPI_bcast(...) calls, see below.
    assert(value == comm.root());

    // You can also specify a custom root for this opertion:
    size_t custom_root_rank = 1;
    value                   = comm.rank();
    comm.bcast(send_recv_buf(value), root(custom_root_rank)); // Two MPI_bcast(...) calls, see below.
    assert(value == custom_root_rank);

    // To broadcast all values in a container, use the following:
    std::vector<int> values(10);
    std::fill(values.begin(), values.end(), comm.rank());
    comm.bcast(send_recv_buf(values));
    for ([[maybe_unused]] int elem: values) {
        assert(elem == comm.root_signed());
    }
    // KaMPIng will broadcast the whole content of the container. The containers on the receiving ranks will be
    // automatically resized to hold exactly the number of elements received.

    // All of these examples so far actually use two MPI_Bcast calls. The first one is needed to communicate how many
    // elements we want to communicate in the second broadcast. This also applies to the single-element version above,
    // as the receiving ranks have no way of knowing if a single element of multiple elements are to be broadcasted. If
    // you already know, how many elements you will receive, you can specify this using the recv_count() parameter. If
    // this parameter is provided on any rank, it has to be provided and be the same on all ranks, including the root!
    // This is, because each rank needs to know whether to broadcast the recv_count or not.
    comm.bcast(send_recv_buf(value), recv_count(1));   // One MPI_bcast(...) call :-)
    comm.bcast(send_recv_buf(values), recv_count(10)); // One MPI_bcast(...) call :-)

    // If you want to know the number of elements received, you can use recv_count as an output parameter:
    int recv_count; // The type is a remnant of MPI's C-style interface.
    comm.bcast(send_recv_buf(values), recv_count_out(recv_count));
    assert(static_cast<size_t>(recv_count) == values.size());
    // The recv_count parameter does not have to be provided on all ranks. It is, however, not allowed to provide the
    // recv_count_out parameter on some ranks and the recv_count parameeter on other ranks in the same call to .bast(),
    // as you have to provide recv_count either on all or on none of the ranks (see above).

    // I you want to do a partial transfer of a container, you have to use a kamping::Span<>
    size_t                                      num_transferred_values = 3;
    kamping::Span<decltype(values)::value_type> transfer_view(
        values.data(), asserting_cast<size_t>(num_transferred_values));
    comm.bcast(send_recv_buf(transfer_view)); // Will broadcast the first three elements in the values container.
    // Using a kamping::Span<> also prevents kamping from resizing the receive buffers, which thus have to be large
    // enough to hold all the received elements.
}
