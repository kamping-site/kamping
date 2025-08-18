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
#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/data_buffers/empty_db.hpp"
#include "kamping/data_buffers/resizable_db.hpp"
#include "kamping/environment.hpp"

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    /*{
        std::vector<int> sbuf(comm.size() + 10, comm.rank_signed() + 5);
        auto rbuf = EmptyDataBuffer<int>();

        // Fails because rbuf is not large enough
        auto [sent, received] = comm.allgather(sbuf, rbuf);
    }*/

    comm.barrier();

    {
        std::vector<int> sbuf(comm.size() + 10, comm.rank_signed() + 5);
        auto             rbuf = ResizeableDataBuffer(EmptyDataBuffer<int>());
        auto [sent, received] = comm.allgather(sbuf, rbuf);

        if (comm.rank() == 0) {
            for (auto x: received) {
                print_on_root(std::to_string(x), comm);
            }
        }
    }

    comm.barrier();
    print_on_root("-----", comm);

    {
        size_t           size = comm.size() + 10;
        std::vector<int> sbuf(size, comm.rank_signed() + 5);
        std::vector<int> rbuf(size * comm.size());
        auto [sent, received] = comm.allgather(sbuf, rbuf);

        if (comm.rank() == 0) {
            for (auto x: received) {
                print_on_root(std::to_string(x), comm);
            }
        }
    }
}
