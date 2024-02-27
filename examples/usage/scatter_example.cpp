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

#include <numeric>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"

int main(int argc, char* argv[]) {
    using namespace kamping;

    Environment e(argc, argv);

    Communicator     comm;
    std::vector<int> input(static_cast<std::size_t>(comm.size()));

    std::iota(input.begin(), input.end(), 0);

    {
        // simply return received data
        auto output = comm.scatter(send_buf(input));
        print_result_on_root(output, comm);
    }
    {
        // write received data to exisiting container
        std::vector<int> output;
        comm.scatter(send_buf(input), recv_buf<resize_to_fit>(output));
        print_result_on_root(output, comm);
    }

    return 0;
}
