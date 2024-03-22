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
#include <random>
#include <unordered_set>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugin/alltoall_dispatch.hpp"
#include "kamping/plugin/alltoall_sparse.hpp"
#include "kamping/span.hpp"

// Helper function used in the example below.
// Generates num_partners distinct random communication partners \in [0,comm_size).
auto random_comm_partners(int comm_size, size_t num_partners) {
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, comm_size - 1);
    std::unordered_set<int>         comm_partners;

    while (comm_partners.size() < num_partners) {
        comm_partners.insert(dis(gen));
    }

    return comm_partners;
}

int main() {
    using namespace kamping;
    using namespace plugin;
    using namespace dispatch_alltoall;

    // Enable the SparseAlltoall, GridCommunicator and DispatchAlltoall plugins.
    using Comm = Communicator<std::vector, plugin::SparseAlltoall, plugin::GridCommunicator, plugin::DispatchAlltoall>;

    kamping::Environment e;
    Comm                 comm;

    { // KaMPIng wraps the classic MPI alltoallv
        // Rank i sends i values to rank 0, i+1 values to rank 1, ...
        std::vector<int> counts_per_rank(comm.size());
        std::iota(counts_per_rank.begin(), counts_per_rank.end(), comm.rank_signed());

        int                 num_elements = std::reduce(counts_per_rank.begin(), counts_per_rank.end(), 0);
        std::vector<size_t> input(asserting_cast<size_t>(num_elements));
        // Rank i sends it own rank to all others
        std::fill(input.begin(), input.end(), comm.rank());

        { // Exchange the data; compute the recv counts automatically.
            auto output = comm.alltoallv(send_buf(input), send_counts(counts_per_rank));
            print_on_root(" --- alltoallv I --- ", comm);
            print_result_on_root(output, comm);
        }

        { // Exchange the data; compute /and return/ the recv counts automatically.
            auto [output, receive_counts] =
                comm.alltoallv(send_buf(input), send_counts(counts_per_rank), recv_counts_out());
            print_on_root(" --- alltoallv II output --- ", comm);
            print_result_on_root(output, comm);
            print_on_root(" --- alltoallv II receive counts --- ", comm);
            print_result_on_root(receive_counts, comm);
        }
    }

    { // For sparse messages exchanges, KaMPIng provides a specialized algorithm via the SparseAlltoall plugin.
        // Generate the messages
        using msg_type = std::vector<double>;

        std::unordered_map<int, msg_type> dest_msg_pairs;
        for (auto const dst: random_comm_partners(comm.size_signed(), comm.size() / 2)) {
            msg_type msg(comm.rank(), static_cast<double>(comm.rank()));
            dest_msg_pairs.emplace(dst, std::move(msg));
        }

        std::unordered_map<int, std::vector<double>> recv_buf;

        // Define a callback function to receive messages.
        auto cb = [&](auto const& probed_message) {
            recv_buf[probed_message.source_signed()] = probed_message.recv();
        };

        // Exchange the messages
        comm.alltoallv_sparse(
            plugin::sparse_alltoall::sparse_send_buf(dest_msg_pairs),
            plugin::sparse_alltoall::on_message(cb)
        );

        print_on_root(" --- Sparse alltoallv --- ", comm);
        if (comm.is_root()) {
            for (auto const& [source, msg]: recv_buf) {
                print_result("source: " + std::to_string(source), comm);
                print_result(msg, comm);
                print_result("---", comm);
            }
        }
    }

    { // The DispatchAlltoall plugin decides whether to use the GridCommunicator or the builtin alltoallv depending on
      // the size and number of messages.
        std::vector<double> const data(comm.size(), static_cast<double>(comm.rank()));
        std::vector<int> const    counts(comm.size(), 1);

        { // Use the provided default thresholds.
            [[maybe_unused]] auto [output, receive_counts] =
                comm.alltoallv_dispatch(send_buf(data), send_counts(counts), recv_counts_out());
        }

        { // Set a custom threshold for the maximum bottleneck send communication volume for when to switch from
            // grid to builtin alltoall.
            [[maybe_unused]] auto [output, receive_counts] = comm.alltoallv_dispatch(
                send_buf(data),
                send_counts(counts),
                comm_volume_threshold(10),
                recv_counts_out()
            );
        }
    }

    return EXIT_SUCCESS;
}
