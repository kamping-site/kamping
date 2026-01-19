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

#include "../../tests/helpers_for_testing.hpp"
#include "helpers_for_examples.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffers/displs_pipes.hpp"
#include "kamping/data_buffers/extended_db.hpp"
#include "kamping/data_buffers/pipe_db.hpp"
#include "kamping/data_buffers/size_v_pipes.hpp"
#include "kamping/environment.hpp"

template <typename T>
void printType() {
    std::cout << __PRETTY_FUNCTION__ << '\n';
}

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

    std::vector<int> recv_size_v = recv_counts;

    kamping_recv_buf.set_size_v(std::move(recv_counts));

    std::vector<int> displs_to_set;

    {
        // FAILED ASSERTION: Displs are not large enough, and resize is not enabled
        // auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | auto_displs(displs_to_set) |
        // resize_ext());
    }

    {
        // Works, displs_to_set is set, received is
        // resize_ext_view<auto_displs_view<kamping::BufferResizePolicy::resize_to_fit,
        // kamping::ranges::kamping_ref_view<ExtDataBuffer>, kamping::ranges::kamping_ref_view<std::vector<int>>>>
        auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf |
        auto_displs<BufferResizePolicy::resize_to_fit>(displs_to_set) | resize_ext());
    }


    {
        // Works, received is resize_ext_view<auto_displs_view<kamping::BufferResizePolicy::resize_to_fit,
        // kamping::ranges::kamping_ref_view<ExtDataBuffer>, kamping::ranges::kamping_owning_view<std::vector<int>>>>
        auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | auto_displs() | resize_ext());
    }


    {
        // Works, received is resize_ext_view<auto_displs_view<kamping::BufferResizePolicy::resize_to_fit,
        // kamping::ranges::kamping_ref_view<ExtDataBuffer>,
        // kamping::ranges::kamping_owning_view<kamping::example_IntRange>>>
        auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | auto_displs<example_IntRange>() |
        resize_ext());
    }

    {
        // FAILED ASSERTION: Displs are not large enough, and resize is not enabled
        // auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | auto_displs(std::move(displs_to_set))
        // | resize_ext());
    }

    {
        // Works, displs_to_set is set, received is
        // resize_ext_view<auto_displs_view<kamping::BufferResizePolicy::resize_to_fit,
        // kamping::ranges::kamping_ref_view<ExtDataBuffer>, kamping::ranges::kamping_ref_view<std::vector<int>>>>
        auto [sent, received] = comm.alltoallv(
            kamping_send_buf,
            kamping_recv_buf | auto_displs<BufferResizePolicy::resize_to_fit>(displs_to_set) | resize_ext()
        );
    }

    {
        // Works, recv_displs are empty, received is
        // resize_ext_view<with_displs_view<kamping::ranges::kamping_ref_view<ExtDataBuffer>, kamping::ranges::kamping_owning_view<std::vector<int>>>>
        auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf | with_displs(std::move(recv_displs)) |
        resize_ext());

    }


    {
        // Works, displs_to_set are empty, received is
        // resize_ext_view<auto_displs_view<kamping::BufferResizePolicy::resize_to_fit,
        // kamping::ranges::kamping_ref_view<ExtDataBuffer>, kamping::ranges::kamping_owning_view<std::vector<int>>>>
        auto [sent, received] = comm.alltoallv(kamping_send_buf, kamping_recv_buf |
        auto_displs<BufferResizePolicy::resize_to_fit>(std::move(displs_to_set)) | resize_ext());
    }



    {
        auto [sent, received] = comm.alltoallv(kamping_send_buf,
        make_auto_displs_view<BufferResizePolicy::resize_to_fit>(kamping_recv_buf, std::move(displs_to_set)));
    }

    {
        auto [sent, received] = comm.alltoallv(kamping_send_buf,
        make_auto_displs_view<BufferResizePolicy::resize_to_fit>(kamping_recv_buf, displs_to_set));
    }

    {
        testing::NonCopyableOwnContainer<int> copy_test(100);

        auto [sent, received] = comm.alltoallv(
        kamping_send_buf, std::move(copy_test) | with_size_v(recv_size_v) | auto_displs());
        auto copy_test_out = std::move(received.get_base());
        // CopyContainer has been moved, this CopyContainer is invalid

        auto copy_invalid_out = std::move(received.get_base());
    }

    {
        testing::NonCopyableOwnContainer<int> copy_test(100);

        auto [sent, received] = comm.alltoallv(
        kamping_send_buf, copy_test | with_size_v(recv_size_v) | auto_displs());

        auto& copy_test_out = received.get_base();
    }

    auto [sent, received] = comm.alltoallv(kamping_send_buf, std::vector<int>(50) | with_size_v(recv_size_v) | auto_displs());
    // Copies
    auto vec_out = received.get_base();
    //auto vec_out_move = std::move(received.get_base());


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

    // Print received.displs()
    auto& x = received.displs();
    if (comm.rank_signed() == 0) {
        printType<decltype(x)>();
        printType<decltype(received)>();
    }
    for (int p = 0; p < static_cast<int>(size); ++p) {
        if (p == rank) {
            std::cout << "Process " << rank << " received.displs():";
            for (auto d: x) {
                std::cout << " " << d;
            }
            std::cout << std::endl;
        }
        comm.barrier();
    }

    return 0;
}
