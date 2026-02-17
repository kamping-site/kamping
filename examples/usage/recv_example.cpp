// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
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
#include <vector>
#include <mdspan>
#include <random>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/adapter/mdspan_adapter.hpp"
#include "kamping/data_buffers/type_pipes.hpp"
#include "kamping/data_buffers/resize_pipes.hpp"

struct example_struct {
    int foo;
    double bar;
};

MPI_Datatype example_type() {
    const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
    MPI_Aint offsets[2];

    offsets[0] = offsetof(example_struct, foo);
    offsets[1] = offsetof(example_struct, bar);

    MPI_Datatype example_type;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &example_type);
    MPI_Type_commit(&example_type);
    return example_type;
}

int main() {
    using namespace kamping;
    kamping::Environment  e;
    kamping::Communicator comm;

    KASSERT(comm.size() == 2uz, "This example must be run with exactly 2 ranks.");

    {
        if (comm.rank_signed() == 0) {
            std::vector<int> data{1, 2, 3, 4, 5, 6, 7, 8, 9};
            std::mdspan<int, std::extents<size_t, 3, 3>> to_send(data.data());
            comm.send(adapter::MDSpanAdapter(to_send), 1);
        }
        else {
            std::vector<int> data(9);
            std::mdspan<int, std::extents<size_t, 3, 3>> to_recv(data.data());
            auto received = comm.recv(adapter::MDSpanAdapter(to_recv), 0);
            auto result = received.get_mdspan();
        }
    }

    {
        if (comm.rank_signed() == 0) {
            std::random_device rd;
            std::mt19937 gen(rd());


            std::uniform_int_distribution<> dist(1, 10);
            int ext1 = dist(gen);
            int ext2 = dist(gen);

            std::vector<int> data(ext1 * ext2, 42);
            std::mdspan<int, std::extents<size_t,  std::dynamic_extent, std::dynamic_extent>> to_send(data.data(), ext1, ext2);
            
            comm.send(std::ranges::single_view(ext1), 1);
            comm.send(std::ranges::single_view(ext2), 1);
            comm.send(adapter::MDSpanAdapter(to_send), 1);
        }
        else {
            std::ranges::single_view v1(0);
            comm.recv(v1, 0);

            std::ranges::single_view v2(0);
            comm.recv(v2, 0);

            int ext1 = v1.front();
            int ext2 = v2.front();

            std::vector<int> data(ext1 * ext2);
            std::mdspan<int, std::extents<size_t, std::dynamic_extent, std::dynamic_extent>> to_recv(data.data(), ext1, ext2);

            auto received = comm.recv(adapter::MDSpanAdapter(to_recv), 0);

            auto result = received.get_mdspan();

        }
    }

    {
        if (comm.rank_signed() == 0) {
            std::vector<example_struct> data(10, {42, 42.42});
            comm.send(data | with_type(example_type()), 1);
        }
        else {
            std::vector<example_struct> data;
            auto result = comm.recv(data | with_type(example_type()) | resize_buf(), 0);
        }
    }
}
