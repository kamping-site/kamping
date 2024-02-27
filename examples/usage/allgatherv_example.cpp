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

    {
        // simply return the recv buffer
        auto recv_buffer = comm.allgatherv(send_buf(input));
        print_result(recv_buffer, comm);
    }

    {
        // return recv buffer and recv_counts
        auto [recv_buffer, recv_counts] = comm.allgatherv(send_buf(input), recv_counts_out());
        print_result(recv_buffer, comm);
        print_result(recv_counts, comm);
    }

    {
        // write result to an exisiting container
        std::vector<int> recv_buffer;
        comm.allgatherv(send_buf(input), recv_buf<resize_to_fit>(recv_buffer));
        print_result(recv_buffer, comm);

        // additionally, receive counts and/or receive displacements can be provided
        std::vector<int> recv_counts(comm.size());
        std::iota(recv_counts.begin(), recv_counts.end(), 0);
        std::vector<int> recv_displs(comm.size());
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);
        recv_buffer.clear();

        comm.allgatherv(
            send_buf(input),
            recv_buf<resize_to_fit>(recv_buffer),
            kamping::recv_counts(recv_counts),
            kamping::recv_displs(recv_displs)
        );
        print_result(recv_buffer, comm);

        return 0;
    }
}
