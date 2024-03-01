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
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <mpi.h>

#include "./kamping.hpp"
#include "./mpi.hpp"

template <typename T>
bool globally_sorted(MPI_Comm comm, std::vector<T> const& data, std::vector<T>& original_data) {
    kamping::Communicator kamping_comm(comm);
    auto                  global_data = kamping_comm.gatherv(kamping::send_buf(data)).extract_recv_buffer();
    auto global_data_original         = kamping_comm.gatherv(kamping::send_buf(original_data)).extract_recv_buffer();
    std::sort(global_data_original.begin(), global_data_original.end());
    return global_data_original == global_data;
}

template <typename T>
auto generate_data(size_t n_local, seed_type seed) -> std::vector<T> {
    std::mt19937                     eng(seed + static_cast<seed_type>(kamping::world_rank()));
    std::uniform_int_distribution<T> dist(0, std::numeric_limits<T>::max());
    std::vector<T>                   data(n_local);
    auto                             gen = [&] {
        return dist(eng);
    };
    std::generate(data.begin(), data.end(), gen);
    return data;
}

int main(int argc, char* argv[]) {
    kamping::Environment env;
    size_t               n_local;
    seed_type            seed = 42;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n_local> [seed]" << std::endl;
        kamping::comm_world().abort();
        return 1;
    }
    std::stringstream ss(argv[1]);
    ss >> n_local;
    if (argc > 2) {
        ss.str(argv[2]);
        ss >> seed;
    }
    using element_type = uint64_t;
    seed_type local_seed =
        seed + static_cast<seed_type>(kamping::world_rank()) + static_cast<seed_type>(kamping::world_size());
    {
        std::vector<element_type> data = generate_data<element_type>(n_local, seed);
        kamping::sort(MPI_COMM_WORLD, data, local_seed);
    }
    {
        std::vector<element_type> data = generate_data<element_type>(n_local, seed);
        mpi::sort(MPI_COMM_WORLD, data, local_seed);
    }
    return 0;
}
