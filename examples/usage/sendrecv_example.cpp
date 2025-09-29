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
#include "kamping/communicator.hpp"
#include "kamping/data_buffers/empty_db.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/sendrecv.hpp"

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    {
        auto   dest = comm.rank_shifted_cyclic(1);
        size_t size = comm.size() + 10;

        std::vector<int> sbuf(size, comm.rank_signed() + 5);
        auto             rbuf = EmptyDataBuffer<int>();
        auto [sent, received] = comm.sendrecv(sbuf, rbuf, static_cast<int>(dest));

        if (comm.rank() == 0) {
            for (auto x: received) {
                print_on_root(std::to_string(x), comm);
            }
        }
    }
}