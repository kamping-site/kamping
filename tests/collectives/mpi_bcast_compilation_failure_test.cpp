// This file is part of KaMPIng.
//
// Copyright 2022-2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    Communicator comm;
    int          value = comm.rank_signed();

#if defined(SEND_RECV_COUNT_GIVEN)
    comm.bcast_single(send_recv_buf(value), send_recv_count(1));
#elif defined(SEND_RECV_TYPE_GIVEN_BUT_NO_SEND_RECV_COUNT)
    comm.bcast(send_recv_buf(value), send_recv_type(MPI_INT));
#elif defined(SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT)
    comm.bcast(send_recv_buf<resize_to_fit>(value), send_recv_type(MPI_INT), send_recv_count(1));
#elif defined(SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY)
    comm.bcast(send_recv_buf<grow_only>(value), send_recv_type(MPI_INT), send_recv_count(1));
#elif defined(SINGLE_VARIANT_WITH_VECTOR)
    std::vector<int> input{value};
    comm.bcast_single(send_recv_buf(input));
#else
    // If none of the above sections is active, this file will compile successfully.
    comm.bcast_single(send_recv_buf(value));
    comm.bcast(send_recv_buf(value), send_recv_type(MPI_INT), send_recv_count(1));
#endif
}
