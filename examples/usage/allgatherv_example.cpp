// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/allgather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"

int main() {
    using namespace kamping;

    // The Environment class is a RAII wrapper around MPI_Init and MPI_Finalize.
    kamping::Environment e;

    // A kamping::Communicator abstracts away an MPI_Comm; here MPI_COMM_WORLD.
    kamping::Communicator comm;

    // Note, that the size of the input vector is different for each rank.
    std::vector<int> input(comm.rank(), comm.rank_signed());

    { // Gather the input from all ranks to the root rank.
        auto recv_buffer = comm.allgatherv(send_buf(input));
    }

    { // We can also request the number of elements received from each rank. The recv_buf will always be the first out
      // parameter. After that, the output parameters are ordered as they are in the function call.
        auto [recv_buffer, recv_counts] = comm.allgatherv(send_buf(input), recv_counts_out());
    }

    { // To re-use memory, we can provide an already allocated container to the MPI call.
        std::vector<int> recv_buffer;
        // Let KaMPIng resize the recv_buffer to the correct size. Other possibilities are no_resize and grow_only.
        comm.allgatherv(send_buf(input), recv_buf<resize_to_fit>(recv_buffer));

        // We can also re-use already allocated containers for the other output parameters, e.g. recv_counts.
        std::vector<int> recv_counts(comm.size());
        std::iota(recv_counts.begin(), recv_counts.end(), 0);
        comm.allgatherv(send_buf(input), recv_buf<resize_to_fit>(recv_buffer), kamping::recv_counts(recv_counts));

        std::vector<int> recv_displs(comm.size());
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);
        recv_buffer.clear();

        // In this example, we combine all of the concepts mentioned above:
        // - Se input as the send buffer
        // - Receive all elements into recv_buffer, resizing it to fit exactly the number of elements received.
        // - Output the number of elements received from each rank into recv_counts.
        // - Output the displacement of the first element received from each rank into recv_displs.
        comm.allgatherv(
            send_buf(input),
            recv_buf<resize_to_fit>(recv_buffer),
            kamping::recv_counts(recv_counts),
            kamping::recv_displs(recv_displs)
        );
    }

    { // If we have many out parameters, we can replace the  structured bindings with extract_*() calls in order to
      // increase readability.
        auto       result      = comm.allgatherv(send_buf(input), recv_counts_out(), recv_displs_out());
        auto const recv_buffer = result.extract_recv_buffer();
        auto const recv_counts = result.extract_recv_counts();
        auto const recv_displs = result.extract_recv_displs();
    }

    return 0;
}
