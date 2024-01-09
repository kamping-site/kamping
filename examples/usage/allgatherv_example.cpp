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
    std::vector<int>      input(comm.rank(), comm.rank_signed());

    // case 1: If no additional output parameters are requested only recv_buffer will be returned
    {
        std::vector<int> recv_buffer = comm.allgatherv(send_buf(input));
        if (comm.is_root())
            print_result(recv_buffer, comm);
    }
    print_on_root("---------------------------", comm);
    // If additional output parameters are requested allgatherv return a result object containing all buffers that are
    // marked as output-buffers (via *_out suffix). Note that the recv_buffer is marked as output buffer implicitly. If
    // the recv_buffer does not own its underlying storage it is not an output buffer and therefore not part of the
    // result object.

    // case 2a.a: the result object can be decomposed with a structured binding (in the order defined in
    // the call, the implicit recv_buffer always comes first).
    {
        auto [recv_buffer, recv_counts, recv_displs] =
            comm.allgatherv(send_buf(input), recv_counts_out(), recv_displs_out());
        if (comm.is_root())
            print_result(recv_buffer, comm);
    }
    print_on_root("---------------------------", comm);
    // case 2a.b: recv_buffer is still an output buffer (as it owns its underlying storage)
    {
        std::vector<int> preallocated_storage(10);
        auto [recv_counts, recv_buffer, recv_displs] = comm.allgatherv(
            send_buf(input),
            recv_counts_out(),
            recv_buf<resize_to_fit>(std::move(preallocated_storage)),
            recv_displs_out()
        );
        if (comm.is_root())
            print_result(recv_buffer, comm);
    }
    // case 2a.c: recv_buffer is not an output buffer
    {
        std::vector<int> recv_buffer;
        auto [recv_counts, recv_displs] = comm.allgatherv(
            send_buf(input),
            recv_counts_out(),
            recv_buf<resize_to_fit>(recv_buffer),
            recv_displs_out()
        );
        // print_result(recv_buffer, comm);
    }

    // case 2b: the buffers within the result object can be retrieved as before
    {
        auto result_obj = comm.allgatherv(send_buf(input), recv_counts_out());
        // print_result(result_obj.extract_recv_buffer(), comm);
        // print_result(result_obj.extract_recv_counts(), comm);
    }

    return 0;
}
