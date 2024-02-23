// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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
#include "kamping/collectives/sparse_alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/span.hpp"

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

    kamping::Environment e;
    Communicator         comm;

    // generate sparse exchange messages
    std::vector<std::pair<int, std::vector<double>>> dst_msg_pairs;
    for (auto const dst: random_comm_partners(comm.size_signed(), comm.size() / 2)) {
        std::vector<double> msg(comm.rank(), static_cast<double>(comm.rank()));
        dst_msg_pairs.emplace_back(dst, std::move(msg));
    }

    std::unordered_map<int, std::vector<double>> recv_buf;
    // prepare callback function to receive messages
    auto on_receive = [&](auto const& probed_message) {
        recv_buf[probed_message.source_signed()] = probed_message.recv();
    };

    comm.alltoallv_sparse(on_receive, kamping::sparse_send_buf(dst_msg_pairs));

    if (comm.is_root()) {
        for (auto const& [source, msg]: recv_buf) {
            print_result("source: " + std::to_string(source), comm);
            print_result(msg, comm);
            print_result("---", comm);
        }
    }

    return 0;
}
