// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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

#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"

int main() {
    using namespace kamping;

    kamping::Environment e;
    Communicator         comm;

    // Rank i sends i values to rank 0, i+1 values to rank 1, ...
    std::vector<int> counts_per_rank(comm.size());
    std::iota(
        counts_per_rank.begin(), counts_per_rank.end(), comm.rank_signed()
    );

    int num_elements =
        std::reduce(counts_per_rank.begin(), counts_per_rank.end(), 0);
    std::vector<size_t> input(asserting_cast<size_t>(num_elements));
    // Rank i sends it own rank to all others
    std::fill(input.begin(), input.end(), comm.rank());

    std::vector<size_t> output =
        comm.alltoallv(send_buf(input), send_counts(counts_per_rank))
            .extract_recv_buffer();

    print_result_on_root(output, comm);

    return 0;
}
