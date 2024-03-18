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
#include "kamping/collectives/allreduce.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

int main(int /*argc*/, char** /*argv*/) {
    using namespace ::kamping;
    Communicator           comm;
    std::vector<int> const input{1};
    std::vector<int>       recv_buffer(1);

#if defined(OPERATION_TYPE_DOES_NOT_MATCH_BUFFER_TYPE)
    auto my_op = [](std::string const& lhs, std::string const&) {
        return lhs;
    };
    comm.allreduce(send_buf(input), op(my_op, kamping::ops::commutative));
#elif defined(SEND_RECV_TYPE_GIVEN_BUT_NO_SEND_RECV_COUNT)
    comm.allreduce(
        send_buf(input),
        send_recv_type(MPI_INT),
        op(kamping::ops::plus<>{}),
        recv_buf<no_resize>(recv_buffer)
    );
#elif defined(SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT)
    comm.allreduce(
        send_buf(input),
        send_recv_type(MPI_INT),
        send_recv_count(1),
        op(kamping::ops::plus<>{}),
        recv_buf<resize_to_fit>(recv_buffer)
    );
#elif defined(SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY)
    comm.allreduce(
        send_buf(input),
        send_recv_type(MPI_INT),
        send_recv_count(1),
        op(kamping::ops::plus<>{}),
        recv_buf<grow_only>(recv_buffer)
    );
#elif defined(SINGLE_VARIANT_WITH_VECTOR)
    int const result = comm.allreduce_single(send_buf(input), op(kamping::ops::plus<>{}));
#else
    // If none of the above sections is active, this file will compile successfully.
    comm.allreduce(
        send_buf(input),
        send_recv_type(MPI_INT),
        send_recv_count(1),
        op(kamping::ops::plus<>{}),
        recv_buf<no_resize>(recv_buffer)
    );
#endif
}
