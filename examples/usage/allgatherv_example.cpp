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

#include <kamping/serialization.hpp>
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

    // The Environment class is a RAII wrapper around MPI_Init and MPI_Finalize.
    kamping::Environment e;

    // A kamping::Communicator abstracts away an MPI_Comm; here MPI_COMM_WORLD.
    kamping::Communicator comm;

    // Note, that the size of the input vector is different for each rank.
    std::vector<int> input(comm.rank(), comm.rank_signed());

    {
        // we are including the namespace here to reduce verbosity
        using namespace kamping;
        { // Basic use case; gather the inputs across all ranks to all ranks.
            auto const output = comm.allgatherv(send_buf(input));
            print_result_on_root(output, comm);
        }

        { // We can also request the number of elements received from each rank. The recv_buf will always be the first
          // out parameter. After that, the output parameters are ordered as they appear in the function call. KaMPIng
          // calls like allgatherv return a result object which can be decomposed using structured bindings (here) or
          // explicit extract_*() calls (see below).
            auto [recv_buffer, recv_counts] = comm.allgatherv(send_buf(input), recv_counts_out());
        }

        { // To re-use memory, we can provide an already allocated container to the MPI call.
            std::vector<int> output;
            // Let KaMPIng resize the recv_buffer to the correct size. Other possibilities are no_resize and grow_only.
            comm.allgatherv(send_buf(input), recv_buf<resize_to_fit>(output));

            // We can also re-use already allocated containers for the other output parameters, e.g. recv_counts.
            std::vector<int> output_counts(comm.size());
            std::iota(output_counts.begin(), output_counts.end(), 0);
            comm.allgatherv(send_buf(input), recv_buf<resize_to_fit>(output), recv_counts(output_counts));

            std::vector<int> displacements(comm.size());
            std::exclusive_scan(output_counts.begin(), output_counts.end(), displacements.begin(), 0);
            output.clear();

            // In this example, we combine all of the concepts mentioned above:
            // - Use input as the send buffer
            // - Receive all elements into recv_buffer, resizing it to fit exactly the number of elements received.
            // - Output the number of elements received from each rank into recv_counts.
            // - Output the displacement of the first element received from each rank into recv_displs.
            comm.allgatherv(
                send_buf(input),
                recv_buf<resize_to_fit>(output),
                recv_counts(output_counts),
                recv_displs(displacements)
            );
        }

        { // It is also possible to use result.extract_*() calls instead of decomposing the result object using
          // structured bindings in order to increase readability.
            auto       result      = comm.allgatherv(send_buf(input), recv_counts_out(), recv_displs_out());
            auto const recv_buffer = result.extract_recv_buffer();
            auto const recv_counts = result.extract_recv_counts();
            auto const recv_displs = result.extract_recv_displs();
        }

        { // C++ views can be used to send parts of the data.
            input.resize(comm.rank() + 1, comm.rank_signed());

            // Note, if you're on C++ >= 20 you can use std::span instead.
            comm.allgatherv(send_buf(kamping::Span(input).subspan(0, comm.rank())));

            // Alternatively
            comm.allgatherv(send_buf(input), send_count(comm.rank_signed()));

            // let's restore the input
            input.resize(comm.rank());
        }
    }
    {
        // if you don't want to polute your global namespace with all KaMPIng's symbols, you can just use the namespace
        // for named parameters
        using namespace kamping::params;
        auto const output = comm.allgatherv(send_buf(input));
        print_result_on_root(output, comm);
    }
    {
        // you can also define a shorthand for the named parameter namespace, to reduce code verbosity
        namespace kmp     = kamping::params;
        auto const output = comm.allgatherv(kmp::send_buf(input));
        print_result_on_root(output, comm);
    }

    return 0;
}
