// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <mpi.h>

#include <random>

#include "kamping/kassert.hpp"

namespace plain_mpi {
int get_pivot(int item, MPI_Comm comm, int size, int seed) {
    int pivot = item;

    std::mt19937                    gen(static_cast<unsigned long>(seed));
    std::uniform_int_distribution<> distribution(0, size - 1);

    int pivot_pe = distribution(gen);
    MPI_Bcast(&pivot, 1, MPI_INT, pivot_pe, comm);
    return pivot;
}

void count(int value, int* sum, int* all_sum, MPI_Comm comm, int size) {
    MPI_Scan(&value, sum, 1, MPI_INT, MPI_SUM, comm);
    *all_sum = *sum;
    MPI_Bcast(all_sum, 1, MPI_INT, size - 1, comm);
}

int quick_sort(int item, MPI_Comm comm, int seed = 0) { // NOLINT
    int size;
    MPI_Comm_size(comm, &size);

    if (size == 1) {
        return item;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    int pivot = get_pivot(item, comm, size, seed);

    int small, all_small;
    count(item < pivot, &small, &all_small, comm, size);

    if (item < pivot) {
        MPI_Bsend(&item, 1, MPI_INT, small - 1, 8, comm);
    } else {
        MPI_Bsend(&item, 1, MPI_INT, all_small + rank - small, 8, comm);
    }

    MPI_Recv(&item, 1, MPI_INT, MPI_ANY_SOURCE, 8, comm, MPI_STATUS_IGNORE);

    MPI_Comm new_comm;
    MPI_Comm_split(comm, rank < all_small, 0, &new_comm);

    return quick_sort(item, new_comm, seed + 1);
}
} // namespace plain_mpi

namespace kamping {
int quick_sort(int item, MPI_Comm comm, int seed = 0) {
    return item; // TODO
}
} // namespace kamping

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    KASSERT(plain_mpi::quick_sort(rank, MPI_COMM_WORLD) == rank);
    KASSERT(plain_mpi::quick_sort(size - 1 - rank, MPI_COMM_WORLD) == rank);

    KASSERT(kamping::quick_sort(rank, MPI_COMM_WORLD) == rank);
    KASSERT(kamping::quick_sort(size - 1 - rank, MPI_COMM_WORLD) == rank);

    MPI_Finalize();
}