// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <iostream>
#include <numeric>
#include <vector>

#include "helpers_for_examples.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;

int main() {
    using namespace kamping;

    kamping::Environment  e;
    kamping::Communicator comm;

    // Broadcast `value` from the root rank to all other ranks.
    size_t value = comm.rank();
    comm.bcast(send_recv_buf(value));

    // Broadcast a vector of values from the root rank to all other ranks. If we do not provide `send_recv_count`,
    // KaMPIng automatically performs the second broadcast necessary to provideall ranks with the correct receive count.
    // This is useful if some ranks do not know how many elements they will receive.
    // Additionally, use rank 1 as the root rank here.
    std::vector<int> values(4);
    std::fill(values.begin(), values.end(), comm.rank());
    comm.bcast(send_recv_buf(values), send_recv_count(4), root(1));
}
