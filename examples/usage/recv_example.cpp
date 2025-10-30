// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
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
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffers/empty_db.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    KASSERT(comm.size() == 2uz, "This example must be run with exactly 2 ranks.");

    {
        size_t           size = comm.size() + 10;
        std::vector<int> sbuf(size, comm.rank_signed() + 5);
        auto             rbuf = EmptyDataBuffer<int>();

        if (comm.rank_signed() == 0) {
            comm.send(destination(1), send_buf(sbuf));
        } else {
            auto received = comm.recv(rbuf, 0);

            for (auto x: received) {
                std::cout << std::to_string(x) << std::endl;
            }
        }
    }
}
