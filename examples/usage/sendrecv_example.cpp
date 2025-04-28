// This file is part of KaMPI.ng.
//
// Copyright 2025 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <iostream>

#include "helpers_for_examples.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/p2p/sendrecv.hpp"

int main() {
    using namespace kamping;

    kamping::Environment  e;
    kamping::Communicator comm;

    std::vector<int> input(1, comm.rank_signed());
    std::vector<int> message;
    auto             dest = comm.rank_shifted_cyclic(1);

    {
        // Cyclic sendrecv given a recv buffer
        comm.sendrecv(
            send_buf(input),
            destination(dest),
            recv_buf<kamping::BufferResizePolicy::resize_to_fit>(message)
        );
        std::cout << "Rank: " << comm.rank_signed() << " Received: " << message[0] << "\n";
    }
    comm.barrier();
    print_on_root("------", comm);

    {
        // Cyclic sendrecv without an explicit recv buffer
        auto received = comm.sendrecv<int>(send_buf(input), destination(dest));
        std::cout << "Rank: " << comm.rank_signed() << " Received: " << received[0] << "\n";
    }
}