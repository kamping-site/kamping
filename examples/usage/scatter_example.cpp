// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation, either version 3 of
// the License, or (at your option) any later version. KaMPIng is distributed in the hope that it
// will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If
// not, see <https://www.gnu.org/licenses/>.

#include <numeric>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"

int main(int argc, char* argv[]) {
    using namespace kamping;

    Environment e(argc, argv);

    Communicator     comm;
    std::vector<int> in(static_cast<std::size_t>(comm.size()));
    std::vector<int> out;

    std::iota(in.begin(), in.end(), 0);

    comm.scatter(send_buf(in), recv_buf(out));
    print_result(out, comm);

    return 0;
}
