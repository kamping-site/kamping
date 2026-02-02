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
#include "kamping/data_buffers/pipes.hpp"
#include "kamping/data_buffers/size_v_pipes.hpp"
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

    // Send i+rank+5 elements to process i
    int total_send = 0;
    for (int i = 0; i < size; ++i) {
        send_counts[i] = i + rank + 5;
        send_displs[i] = total_send;
        total_send += send_counts[i];
    }

    std::vector<int> send_buf(total_send, rank);

    int total_recv = 0;
    for (int i = 0; i < size; ++i) {
        recv_counts[i] = i + rank + 5;
        recv_displs[i] = total_recv;
        total_recv += recv_counts[i];
    }
    // Prepare send buffer with correct size_v and displs
    auto sbuf = ExtDataBuffer(send_buf);
    sbuf.set_size_v(std::move(send_counts));
    sbuf.set_displs(std::move(send_displs));

    // The following shows how to use kamping's pipes with alltoallv given a send buffer that holds size_v and displs
    {
        // The most basic approach: use a vector and pipe the needed size_v, displs and resizing of the buffer.
        std::vector<int> recv_buf;
        auto [sent, received] =
            comm.alltoallv(sbuf, recv_buf | with_size_v(recv_counts) | auto_displs() | resize_ext());
        // The computed displs can be accessed via
        auto& displs = received.displs();
    }

    {
        // Same as before using the convenience pipe make_vbuf
        std::vector<int> recv_buf;
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf(recv_buf, recv_counts));
    }

    {
        // Using the convenience pipe make_vbuf with an inplace constructed buffer
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf(std::vector<int>(), recv_counts));

        auto result = received.extract_buffer();
    }

    {
        // Using the convenience pipe make_vbuf_vector to implicitly create a buffer of the given type
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf_vector<int>(recv_counts));

        auto result = received.extract_buffer();
    }

    {
        using namespace kamping::pipes;
        // One-liner using make_vbuf_vector
        auto result = comm.alltoallv(sbuf, make_vbuf_vector<int>(recv_counts)).second.extract_buffer();
    }

    {
        // The recv buffer can be moved into the pipe
        std::vector<int> recv_buf;
        auto [sent, received] = comm.alltoallv(sbuf, kamping::pipes::make_vbuf(std::move(recv_buf), recv_counts));

        // A ref to the recv buffer can be retrieved using:
        auto& recv_buf_out = received.buffer();

        // The recv buffer can also be moved out using:
        auto recv_buf_out_move = received.extract_buffer();
    }

    {
        // The type of the displs can be user defined:
        std::vector<int> recv_buf;
        auto [sent, received] =
            comm.alltoallv(sbuf, recv_buf | with_size_v(recv_counts) | auto_displs<example_IntRange>() | resize_ext());

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
            recv_buf | with_size_v(recv_counts) | auto_displs<BufferResizePolicy::resize_to_fit>(displs) | resize_ext()
        );
    }

    {
        // If the displs are known, they can be directly set using with_displs:
        std::vector<int> recv_buf;
        auto [sent, received] =
            comm.alltoallv(sbuf, recv_buf | with_size_v(recv_counts) | with_displs(recv_displs) | resize_ext());
    }

    {
        // If the displs are known, they can be moved into with_displs:
        std::vector<int> recv_buf;
        auto [sent, received] = comm.alltoallv(
            sbuf,
            recv_buf | with_size_v(recv_counts) | with_displs(std::move(recv_displs)) | resize_ext()
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
