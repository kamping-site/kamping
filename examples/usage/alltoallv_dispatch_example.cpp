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
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugin/alltoall_dispatch.hpp"
#include "kamping/span.hpp"

int main() {
    using namespace kamping;
    using namespace plugin;
    using namespace dispatch_alltoall;

    kamping::Environment e;
    using Comm = Communicator<std::vector, plugin::SparseAlltoall, plugin::GridCommunicator, plugin::DispatchAlltoall>;
    Comm comm;

    const std::vector<double> data(comm.size(), static_cast<double>(comm.rank()));
    const std::vector<int>    counts(comm.size(), 1);
    {
        // the plugin decides whether to use grid or builtin alltoall
        auto [recv_buf, recv_counts] = comm.alltoallv_dispatch(send_buf(data), send_counts(counts), recv_counts_out());
    }
    {
        // set another threshold for the maximum bottleneck send communication volume for when to switch from grid to
        // builtin alltoall
        auto [recv_buf, recv_counts] =
            comm.alltoallv_dispatch(send_buf(data), send_counts(counts), comm_volume_threshold(10), recv_counts_out());
    }
    return 0;
}
