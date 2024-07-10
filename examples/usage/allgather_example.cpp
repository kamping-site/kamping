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
    // std::vector<int>      input(comm.size(), comm.rank_signed());

    //{ // Basic form: Provide a send buffer and let KaMPIng allocate the receive buffer.
    //    auto output = comm.allgather(send_buf(input));
    //    print_result_on_root(output, comm);
    //}

    // print_on_root("------", comm);

    //{ // We can also send only parts of the input and specify an explicit receive buffer.
    //    std::vector<int> output;

    //    // this can also be achieved with `kamping::Span`
    //    comm.allgather(send_buf(Span(input.begin(), 2)), recv_buf<resize_to_fit>(output));
    //    print_result_on_root(output, comm);
    //    return 0;
    //}

    MPI_Comm         mpi_graph_comm;
    int              in_degree = -1, out_degree = -1;
    std::vector<int> input{comm.rank_signed()};
    std::vector<int> recv_buf(10, -1);
    int              send_count = -1, recv_count = -1;
    std::vector<int> in_ranks, out_ranks;
    if (comm.rank() == 0) {
        in_degree  = 0;
        out_degree = 0;
        send_count = 1;
        recv_count = 1;
    } else if (comm.rank() == 1) {
        in_degree  = 0;
        out_degree = 1;
        out_ranks.push_back(2);
        send_count = 1;
        recv_count = 1;
    } else if (comm.rank() == 2) {
        in_degree  = 1;
        out_degree = 1;
        in_ranks.push_back(1);
        out_ranks.push_back(3);
        send_count = 1;
        recv_count = 1;
    } else if (comm.rank() == 3) {
        in_degree  = 1;
        out_degree = 0;
        in_ranks.push_back(2);
        send_count = 1;
        recv_count = 1;
    }
    MPI_Dist_graph_create_adjacent(
        comm.mpi_communicator(),
        in_degree,
        in_ranks.data(),
        MPI_UNWEIGHTED,
        out_degree,
        out_ranks.data(),
        MPI_UNWEIGHTED,
        MPI_INFO_NULL,
        false,
        &mpi_graph_comm
    );

    MPI_Neighbor_alltoall(input.data(), send_count, MPI_INT, recv_buf.data(), recv_count, MPI_INT, mpi_graph_comm);
}
