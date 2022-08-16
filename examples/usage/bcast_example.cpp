// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPI.ng is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
// General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPI.ng.  If not, see <https://www.gnu.org/licenses/>.

#include <iostream>
#include <numeric>
#include <vector>

#include "helpers_for_examples.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/parameter_factories.hpp"

using namespace ::kamping;

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    /// @todo Expand these examples, once we have send_recv_buf as unnamed first
    /// parameter.

    size_t value = comm.rank();
    comm.bcast(send_recv_buf(value));
    print_result(value, comm);

    comm.barrier();
    if (comm.is_root()) {
        std::cout << "-------------------" << std::endl;
    }
    comm.barrier();

    value = comm.rank();
    comm.bcast_single(send_recv_buf(value));
    print_result(value, comm);

    comm.barrier();
    if (comm.is_root()) {
        std::cout << "-------------------" << std::endl;
    }
    comm.barrier();

    std::vector<int> values(4);
    std::fill(values.begin(), values.end(), comm.rank());
    comm.bcast(send_recv_buf(values), recv_counts(4), root(1));
    print_result(values, comm);

    // The expected output on 4 ranks is a permutation of the following lines:
    /// @todo Update expected output, once we have the logger which collects
    /// output on the root rank to avoid interleaving output.
    // [PE 0] 0
    // [PE 1] 0
    // [PE 2] 0
    // [PE 3] 0
    // -------------------
    // [PE 0] 0
    // [PE 1] 0
    // [PE 2] 0
    // [PE 3] 0
    // -------------------
    // [PE 0] 1
    // [PE 0] 1
    // [PE 0] 1
    // [PE 0] 1
    // [PE 1] 1
    // [PE 1] 1
    // [PE 1] 1
    // [PE 1] 1
    // [PE 2] 1
    // [PE 2] 1
    // [PE 2] 1
    // [PE 2] 1
    // [PE 3] 1
    // [PE 3] 1
    // [PE 3] 1
    // [PE 3] 1
}
