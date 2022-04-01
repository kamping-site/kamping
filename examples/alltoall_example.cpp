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

#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"

int main() {
    using namespace kamping;

    MPI_Init(NULL, NULL);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    Communicator     comm;
    std::vector<int> input(asserting_cast<size_t>(comm.size()));
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> output;

    comm.alltoall(send_buf(input), recv_buf(output));

    print_result_on_root(output, comm);

    MPI_Finalize();
    return 0;
}
