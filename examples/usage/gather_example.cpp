// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"

int main() {
    using namespace kamping;

    kamping::Environment  e;
    kamping::Communicator comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);

    { // Gather all inputs on rank 0.
        [[maybe_unused]] auto output = comm.gather(send_buf(input));
    }

    { // Receive the gathered data in an existing container on rank 1.
        std::vector<int> output;
        comm.gather(send_buf(input), recv_buf<resize_to_fit>(output), root(0));
    }

    return EXIT_SUCCESS;
}
