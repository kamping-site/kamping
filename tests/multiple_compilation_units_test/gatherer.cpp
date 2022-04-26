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

/// @file
/// @brief The source file for the second compilation unit of a test that checks if compiling and running works
/// correctly when linking two compilation units that both use KaMPI.ng

#include <vector>

// include all collectives so we can catch errors
#include "./gatherer.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/kassert.hpp"

std::vector<int> Gatherer::gather(int data) {
    using namespace kamping;
    KASSERT(mpi_env.initialized());
    Communicator comm;
    auto         result = comm.gather(send_buf(data)).extract_recv_buffer();
    return result;
}
