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
#pragma once
#include <random>

#include "./common.hpp"
#include "kamping/mpi_datatype.hpp"
namespace mpi {
template <typename T>
void sort(MPI_Comm comm, std::vector<T>& data, seed_type seed) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    size_t const   oversampling_ratio = 16 * static_cast<size_t>(std::log2(size)) + 1;
    std::vector<T> local_samples(oversampling_ratio);
    std::sample(data.begin(), data.end(), local_samples.begin(), oversampling_ratio, std::mt19937{seed});
    std::vector<T> global_samples(local_samples.size() * static_cast<size_t>(size));
    MPI_Allgather(
        local_samples.data(),
        static_cast<int>(local_samples.size()),
        kamping::mpi_datatype<T>(),
        global_samples.data(),
        static_cast<int>(local_samples.size()),
        kamping::mpi_datatype<T>(),
        comm
    );

    pick_splitters(static_cast<size_t>(size) - 1, oversampling_ratio, global_samples);
    auto             buckets = build_buckets(data, global_samples);
    std::vector<int> sCounts, sDispls, rCounts(static_cast<size_t>(size)), rDispls(static_cast<size_t>(size));
    size_t           send_pos = 0;
    for (auto& bucket: buckets) {
        data.insert(data.end(), bucket.begin(), bucket.end());
        sCounts.push_back(static_cast<int>(bucket.size()));
        sDispls.push_back(static_cast<int>(send_pos));
        send_pos += bucket.size();
    }
    MPI_Alltoall(sCounts.data(), 1, MPI_INT, rCounts.data(), 1, MPI_INT, comm);

    // exclusive prefix sum of recv displacements
    std::exclusive_scan(rCounts.begin(), rCounts.end(), rDispls.begin(), 0);
    std::vector<T> rData(static_cast<size_t>(rDispls.back() + rCounts.back()));
    MPI_Alltoallv(
        data.data(),
        sCounts.data(),
        sDispls.data(),
        kamping::mpi_datatype<T>(),
        rData.data(),
        rCounts.data(),
        rDispls.data(),
        kamping::mpi_datatype<T>(),
        comm
    );
    std::sort(rData.begin(), rData.end());
    rData.swap(data);
}

} // namespace mpi
