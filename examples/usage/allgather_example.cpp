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
    kamping::Environment  e;
    kamping::Communicator comm;
    std::vector<int>      input(comm.size(), comm.rank_signed());

    { // Basic form: Provide a send buffer and let KaMPIng allocate the receive buffer.
        auto output = comm.allgather(send_buf(input));
        print_result_on_root(output, comm);
    }

    print_on_root("------", comm);

    { // We can also send only parts of the input and specify an explicit receive buffer.
        std::vector<int> output;

        // this can also be achieved with `kamping::Span`
        comm.allgather(send_buf(Span(input.begin(), 2)), recv_buf<resize_to_fit>(output));
        print_result_on_root(output, comm);
        return 0;
    }
}
