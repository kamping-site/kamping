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

#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"

int main() {
    using namespace kamping;

    kamping::Environment  e;
    kamping::Communicator comm;

    std::vector<int> input(comm.rank() + 2, comm.rank_signed());

    { // Gather the data (of different lengths) from all ranks to rank 0
        // simply return received data
        [[maybe_unused]] auto const output = comm.gatherv(send_buf(input), root(0));
    }

    { // Gather the first rank() + 1 elements from all ranks to an existing container on rank 1.
        // Additionally, output the receive counts and displacements.
        std::vector<int> output;

        auto result = comm.gatherv(
            send_buf(input),
            recv_buf<resize_to_fit>(output),
            root(1),
            recv_counts_out(),
            recv_displs_out(),
            send_count(comm.rank_signed() + 1)
        );
        [[maybe_unused]] auto const receive_counts = result.extract_recv_counts();
        [[maybe_unused]] auto const receive_displs = result.extract_recv_displs();
        // Compile time error, as the recv buffer is not owned by KaMPIng, as it was provided via an in parameter.
        // [[maybe_unused]] auto const recv_buf       = result.extract_recv_buffer();
    }

    return 0;
}
