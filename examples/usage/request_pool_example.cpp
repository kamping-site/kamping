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
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/request_pool.hpp"

int main() {
    using namespace kamping;
    Environment  env;
    Communicator comm;
    RequestPool  pool;
    if (comm.rank() == 0) {
        auto req    = pool.get_request();
        auto result = comm.isend(send_buf(42), destination(0), request(std::move(req)));
        std::cout << comm.recv_single<int>() << std::endl;
    }
    auto statuses = pool.wait_all(statuses_out());
}
