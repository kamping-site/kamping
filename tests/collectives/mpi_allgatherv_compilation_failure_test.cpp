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

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/allgather.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    Communicator     comm;
    std::vector<int> input{0};
    std::vector<int> recv_buffer(comm.size());
    std::vector<int> recv_counts_buffer(comm.size(), 1);

#if defined(SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT)
    comm.allgatherv(send_buf(input), send_type(MPI_INT));
#elif defined(RECV_TYPE_GIVEN_BUT_NO_RECV_COUNT)
    comm.allgatherv(send_buf(input), recv_type(MPI_INT), recv_buf<no_resize>(recv_buffer));
#elif defined(RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT)
    comm.allgatherv(
        send_buf(input),
        recv_type(MPI_INT),
        recv_counts(recv_counts_buffer),
        recv_buf<resize_to_fit>(recv_buffer)
    );
#elif defined(RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY)
    comm.allgatherv(
        send_buf(input),
        recv_type(MPI_INT),
        recv_counts(recv_counts_buffer),
        recv_buf<grow_only>(recv_buffer)
    );
#else
    // If none of the above sections is active, this file will compile successfully.
    comm.allgatherv(
        send_buf(input),
        send_type(MPI_INT),
        send_count(1),
        recv_type(MPI_INT),
        recv_counts(recv_counts_buffer),
        recv_buf<no_resize>(recv_buffer)
    );
#endif
}
