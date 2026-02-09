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
#include "kamping/data_buffers/pipes.hpp"
#include "kamping/data_buffers/resize_pipes.hpp"
#include "kamping/data_buffers/size_v_pipes.hpp"
#include "kamping/environment.hpp"

int main() {
    using namespace kamping;

    kamping::Environment  e;
    kamping::Communicator comm;

    int    rank = comm.rank_signed();
    size_t size = comm.size();

    // Preparing send and recv data
    std::vector<int> send_counts(size);
    std::vector<int> recv_counts(size);
    std::vector<int> send_displs(size);
    std::vector<int> recv_displs(size);

    // Send i + rank + 5 elements to process i
    int total_send = 0;
    for (int i = 0; i < size; ++i) {
        send_counts[i] = i + rank + 5;
        send_displs[i] = total_send;
        total_send += send_counts[i];
    }
    std::vector<int> send_vec(total_send, rank);
    int              total_recv = 0;
    for (int i = 0; i < size; ++i) {
        recv_counts[i] = i + rank + 5;
        recv_displs[i] = total_recv;
        total_recv += recv_counts[i];
    }

    // The following shows how to use kamping's pipes with alltoallv
    {
        // The explicit approach, using the previously computed send/recv size_v and displs
        std::vector<int> recv_buf(total_recv);
        auto [sent, received] = comm.alltoallv(
            send_vec | with_size_v(send_counts) | with_displs(send_displs),
            recv_buf | with_size_v(recv_counts) | with_displs(recv_displs)
        );
    }

    {
        // Use the auto_displs pipe to let kamping implicitly compute the displs
        std::vector<int> recv_buf(total_recv);
        auto [sent, received] = comm.alltoallv(
            send_vec | with_size_v(send_counts) | auto_displs(),
            recv_buf | with_size_v(recv_counts) | auto_displs()
        );
        // The computed displs can be accessed via
        auto& displs = received.displs();
    }

    // For simplicity, some of the following examples use this send buffer
    auto sbuf = send_vec | with_size_v(send_counts) | with_displs(send_displs);

    {
        // Use the convenience pipe make_vbuf which is the same as recv_buf | with_size_v(recv_counts) | auto_displs()
        std::vector<int> recv_buf(total_recv);
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf(recv_buf, recv_counts));
    }

    {
        // A more generic approach is to use an empty recv buffer and use the resize_vbuf pipe to resize it accordingly
        std::vector<int> recv_buf;
        auto [sent, received] =
            comm.alltoallv(sbuf, recv_buf | with_size_v(recv_counts) | auto_displs() | resize_vbuf());
    }

    {
        // Use the convenience pipe make_vbuf_resize to implicitly resize the (empty) recv buffer
        std::vector<int> recv_buf;
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf_resize(recv_buf, recv_counts));
    }

    {
        // The recv buffer can be constructed inside the make_vbuf_resize pipe
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf_resize(std::vector<int>(), recv_counts));
        // To retrieve the recv buffer, it can be moved out using extract_buffer:
        auto result = received.extract_buffer();
    }

    {
        // Use the convenience pipe make_vbuf_vector to implicitly create a std::vector of the given type as recv buffer
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf_vector<int>(recv_counts));

        auto result = received.extract_buffer();
    }

    {
        // Kamping can compute the recv_size_v using additional implicit communication. So any recv_size_v of the
        // correct size can be used. Kamping does not support resizing the size_v container, therefore it needs to be
        // large enough
        std::vector<int> recv_buf(total_recv);
        std::vector<int> recv_size_v(comm.size());
        auto [sent, received] = comm.alltoallv(sbuf, recv_buf | with_size_v(recv_size_v) | auto_displs());
    }

    {
        using namespace kamping::pipes;
        // Using the convenience pipe make_vbuf_auto the recv buffer will be a std::vector of the given type.
        // The recv_size_v will be constructed with the given size
        auto result = comm.alltoallv(sbuf, make_vbuf_auto<int>(comm.size())).second.extract_buffer();
    }

    {
        // The recv buffer can be moved into the pipe
        std::vector<int> recv_buf(total_recv);
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf(std::move(recv_buf), recv_counts));

        // A ref to the recv buffer can be retrieved using:
        auto& recv_buf_out = received.buffer();

        // The recv buffer can also be moved out using:
        auto recv_buf_out_move = received.extract_buffer();
    }

    {
        // The type of the computed displs can be user defined:
        std::vector<int> recv_buf;
        auto [sent, received] =
            comm.alltoallv(sbuf, recv_buf | with_size_v(recv_counts) | auto_displs<example_IntRange>() | resize_vbuf());

        // The computed example_IntRange can be accessed via
        auto& displs = received.displs();
    }

    {
        // The displs can be computed into an existing container. Kamping won't resize the displs container by default,
        // so either it is large enough or the ResizePolicy is used
        std::vector<int> recv_buf;
        std::vector<int> displs;
        auto [sent, received] = comm.alltoallv(
            sbuf,
            recv_buf | with_size_v(recv_counts) | auto_displs<BufferResizePolicy::resize_to_fit>(displs) | resize_vbuf()
        );
    }

    {
        // If the displs are known, they can be directly set using with_displs:
        std::vector<int> recv_buf;
        auto [sent, received] =
            comm.alltoallv(sbuf, recv_buf | with_size_v(recv_counts) | with_displs(recv_displs) | resize_vbuf());
    }

    {
        // If the displs are known, they can be moved into with_displs:
        std::vector<int> recv_buf;
        auto [sent, received] = comm.alltoallv(
            sbuf,
            recv_buf | with_size_v(recv_counts) | with_displs(std::move(recv_displs)) | resize_vbuf()
        );

        auto& displs = received.displs();
    }

    {
        // Using a non-copyable container as recv buffer:
        testing::NonCopyableOwnContainer<int> copy_test(100);
        auto [sent, received] = comm.alltoallv(sbuf, std::move(copy_test) | with_size_v(recv_counts) | auto_displs());

        auto copy_test_out = received.extract_buffer();
    }

    return 0;
}
