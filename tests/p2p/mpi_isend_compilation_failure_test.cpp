// This file is part of KaMPI.ng.
//
// Copyright 2023 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "../helpers_for_testing.hpp"
#include "kamping/communicator.hpp"
#include "kamping/p2p/isend.hpp"

using namespace ::kamping;
using namespace ::testing;

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    Communicator comm;
    int          value [[maybe_unused]] = comm.rank_signed();
    int          count                  = 1;

#if defined(SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_STANDARD_MODE)
    comm.isend(send_buf(value), send_type(MPI_INT), destination(comm.rank_shifted_cyclic(1)));
#elif defined(SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_SYNCHRONOUS_MODE)
    comm.issend(send_buf(value), send_type(MPI_INT), destination(comm.rank_shifted_cyclic(1)));
#elif defined(SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_BUFFERED_MODE)
    comm.ibsend(send_buf(value), send_type(MPI_INT), destination(comm.rank_shifted_cyclic(1)));
#elif defined(SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_READY_MODE)
    comm.irsend(send_buf(value), send_type(MPI_INT), destination(comm.rank_shifted_cyclic(1)));
#else
    //    // If none of the above sections is active, this file will compile successfully.
    comm.isend(send_buf(value), send_type(MPI_INT), send_count(count), destination(comm.rank_shifted_cyclic(1)));
    comm.issend(send_buf(value), send_type(MPI_INT), send_count(count), destination(comm.rank_shifted_cyclic(1)));
    comm.ibsend(send_buf(value), send_type(MPI_INT), send_count(count), destination(comm.rank_shifted_cyclic(1)));
    comm.irsend(send_buf(value), send_type(MPI_INT), send_count(count), destination(comm.rank_shifted_cyclic(1)));
#endif
}
