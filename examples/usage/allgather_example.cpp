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
#include "kamping/data_buffers/empty_db.hpp"
#include "kamping/data_buffers/test_db.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    std::vector<int> input(comm.size(), comm.rank_signed() + 5);

    auto [sent, received] = comm.allgather(input, EmptyDataBuffer<int>());

    if (comm.rank() == 0) {
        std::cout << "Input is lvalue: " << std::is_lvalue_reference_v<decltype(sent)> << std::endl;
        std::cout << "Output is lvalue: " << std::is_lvalue_reference_v<decltype(received)> << std::endl;
    }

    print_result_on_root(sent, comm);
    print_on_root("------", comm);

    if (comm.rank() == 0) {
        for (auto x: received) {
            std::cout << x << std::endl;
        }
    }
}
