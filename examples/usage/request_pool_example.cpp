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

#include "kamping/communicator.hpp"
#include "kamping/p2p/irecv.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/request_pool.hpp"

int main() {
    using namespace kamping;
    Environment  env;
    Communicator comm;
    RequestPool  pool;
    if (comm.rank() == 0) {
        for (int i = 0; i < comm.size(); ++i) {
            comm.isend(send_buf(i), destination(i), tag(i), request(pool.get_request()));
        }
    }
    int val;
    comm.irecv(recv_buf(val), request(pool.get_request()));
    auto statuses = pool.wait_all(statuses_out());
    for (MPI_Status& native_status: statuses) {
        Status status(native_status);
        std::cout << "[R" << comm.rank() << "] "
                  << "Status(source="
                  << (status.source_signed() == MPI_PROC_NULL ? "MPI_PROC_NULL" : std::to_string(status.source_signed())
                     )
                  << ", tag=" << (status.tag() == MPI_ANY_TAG ? "MPI_ANY_TAG" : std::to_string(status.tag()))
                  << ", count=" << status.count<int>() << ")" << std::endl;
    }
}
