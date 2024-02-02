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

#include "../helpers_for_testing.hpp"
#include "kamping/communicator.hpp"
#include "kamping/p2p/recv.hpp"

using namespace ::kamping;
using namespace ::testing;

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    Communicator comm;
    int          value [[maybe_unused]] = comm.rank_signed();

#if defined(OWNING_STATUS)
    comm.recv_single<int>(status_out());
#elif defined(PROC_NULL)
    comm.recv_single<int>(source(rank::proc_null));
#elif defined(RECV_COUNT_GIVEN)
    comm.recv_single<int>(recv_counts(1));
#elif defined(RECV_BUF_GIVEN)
    comm.recv_single<int>(recv_buf(value));
#elif defined(RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT)
    ccomm.recv(recv_buf<resize_to_fit>(value), recv_type(MPI_INT));
#elif defined(RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY)
    ccomm.recv(recv_buf<grow_only>(value), recv_type(MPI_INT));
#else
    // If none of the above sections is active, this file will compile successfully.
    comm.recv_single<int>();
    comm.recv(recv_buf<no_resize>(value), recv_type(MPI_INT));
#endif
}
