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
    std::vector<int>      input(comm.rank(), comm.rank_signed());

    {
        // simply return received data
        auto output = comm.gatherv(send_buf(input), root(0));
        print_result_on_root(output, comm);
    }

    {
        // write received data to exisiting container
        std::vector<int> output;
        comm.gatherv(send_buf(input), recv_buf<resize_to_fit>(output), root(0));
        print_result_on_root(output, comm);
    }

    return 0;
}
