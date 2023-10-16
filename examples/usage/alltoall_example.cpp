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
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"

int main() {
    using namespace kamping;

    kamping::Environment e;
    Communicator         comm;
    std::vector<int>     input(2u * comm.size(), comm.rank_signed());
    std::vector<int>     output;

    {
        // send/recv counts are automatically deduced
        comm.alltoall(send_buf(input), recv_buf(output));
        print_result_on_root(output, comm);
        print_on_root("------", comm);
        output.clear();
    }
    {
        // send and recv count can also be explicitly given
        comm.alltoall(send_buf(input), send_count(2), recv_count(2), recv_buf(output));
        print_result_on_root(output, comm);
        print_on_root("------", comm);
        output.clear();

        comm.alltoall(send_buf(input), send_count(1), recv_count(1), recv_buf(output));
        print_result_on_root(output, comm);
        print_on_root("------", comm);
        output.clear();
    }

    return 0;
}
