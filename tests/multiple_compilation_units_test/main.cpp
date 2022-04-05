// This file is part of KaMPI.ng.
//
// Copyright 2021-2022 The KaMPI.ng Authors
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
/// @brief The main file (and first compilation unit) of a test that checks if compiling and running works correctly
/// when linking two compilation units that both use KaMPI.ng

#include <cstddef>

#include <mpi.h>

#include "./gatherer.hpp"
#include "kamping/communicator.hpp"

/// @brief The main function for this tests. Gathers the ranks on the root (done in a different compilation unit), calls
/// a barrier and checks the result
///
/// @param argc The number of command line arguments
/// @param argv The command line arguments
/// @return 0 if successfull
int main(int argc, char** argv) {
    using namespace kamping;

    MPI_Init(&argc, &argv);
    Communicator comm;

    Gatherer gatherer;
    auto     gathered_data = gatherer.gather(comm.rank());

    comm.barrier();

    // Using assert here because this is supposed to do thing the way a normal KaMPI.ng application would do it.
    // Since KASSERT is a KaMPI.ng internal, we cannot use it here.
    if (comm.rank() == 0) {
        assert(gathered_data.size() == static_cast<size_t>(comm.size()));
        for (int rank = 0; rank < comm.size(); ++rank) {
            assert(gathered_data[static_cast<size_t>(rank)] == rank);
        }
    } else {
        assert(gathered_data.empty());
    }

    return 0;
}
