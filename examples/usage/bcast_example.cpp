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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "helpers_for_examples.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/parameter_factories.hpp"

using namespace ::kamping;
using namespace ::testing;

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    /// @todo Expand these examples, once we have send_recv_buf as unnamed first parameter.
    /// @todo Expand these examples, once we have bcast_single.

    // You can broadcast a single element from the communicators root rank to all other ranks using:
    size_t value = comm.rank();
    comm.bcast(send_recv_buf(value));
    print_result(value, comm);

    comm.barrier();
    if (comm.is_root()) {
        std::cout << "-------------------" << std::endl;
    }
    comm.barrier();

    std::vector<int> values(4);
    std::fill(values.begin(), values.end(), comm.rank());
    comm.bcast(send_recv_buf(values), recv_count(4), root(1));
    print_result(values, comm);
}
