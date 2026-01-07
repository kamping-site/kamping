// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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
#include <list>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffers/displs_pipes.hpp"
#include "kamping/data_buffers/extended_db.hpp"
#include "kamping/data_buffers/pipe_db.hpp"
#include "kamping/environment.hpp"

int main() {
    using namespace kamping;

    kamping::Environment  e;
    kamping::Communicator comm;

    int    rank = comm.rank_signed();
    size_t size = comm.size();

    std::vector<int> send_counts(size);
    std::vector<int> recv_counts(size);
    std::vector<int> send_displs(size);
    std::vector<int> recv_displs(size);

    // Send i+rank elements to process i
    int total_send = 0;
    for (size_t i = 0; i < size; ++i) {
        send_counts[i] = (int)i + rank;
        send_displs[i] = total_send;
        total_send += send_counts[i];
    }

    std::vector<int> send_buf((size_t)total_send, rank);

    int total_recv = 0;
    for (size_t i = 0; i < size; ++i) {
        recv_counts[i] = rank + (int)i;
        recv_displs[i] = total_recv;
        total_recv += recv_counts[i];
    }

    std::vector<int> recv_buf(total_recv);

    auto kamping_send_buf = ExtDataBuffer(send_buf);
    auto kamping_recv_buf = ExtDataBuffer(recv_buf);

    kamping_send_buf.set_size_v(std::move(send_counts));
    kamping_send_buf.set_displs(std::move(send_displs));

    kamping_recv_buf.set_size_v(std::move(recv_counts));

    std::vector<int> displs_to_set{1,2};
    // FAILED ASSERTION: Displs are not large enough, and resize is not enabled
    // auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | auto_displs(displs_to_set) |
    // resize_ext());

    // Works, displs_to_set contains computed displs
    // auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf |
    // auto_displs<BufferResizePolicy::resize_to_fit>(displs_to_set) | resize_ext());

    // Works
    // auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | auto_displs() | resize_ext());

    // Works, received.displs() returns a example_IntRange
    // auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | auto_displs<example_IntRange>() |
    // resize_ext());

    // FAILED ASSERTION: Displs are not large enough, and resize is not enabled
    // auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | auto_displs(std::move(displs_to_set))
    // | resize_ext());

    // Works, displs_to_set is empty
    auto [sent, received] = comm.alltoallv(
        kamping_send_buf,
        kamping_recv_buf | auto_displs<BufferResizePolicy::resize_to_fit>(displs_to_set) //| resize_ext()
    );

    //auto ref = displs_to_set;
    //std::ranges::ref_view<decltype(ref)>{ref};

    //auto test = kamping_recv_buf | auto_displs<BufferResizePolicy::resize_to_fit>(std::move(displs_to_set));

    // Works
    //auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | with_displs(std::move(recv_displs)) | resize_ext());

    // Works
    //auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | with_displs(recv_displs) | resize_ext());
    //test.displs();
    //std::vector<int> received(5);

    // Print results
    comm.barrier();
    for (int p = 0; p < static_cast<int>(size); ++p) {
        if (p == rank) {
            std::cout << "Process " << rank << " received:";
            for (int val: received)
                std::cout << " " << val;
            std::cout << std::endl;
        }
        comm.barrier();
    }

    // Print displs_to_set
    comm.barrier();
    for (int p = 0; p < static_cast<int>(size); ++p) {
        if (p == rank) {
            std::cout << "Process " << rank << " recv displs:";
            for (auto d: displs_to_set) {
                std::cout << " " << d;
            }
            std::cout << std::endl;
        }
        comm.barrier();
    }

    auto& x = received.displs();

    return 0;
}
