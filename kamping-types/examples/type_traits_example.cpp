// This file is part of KaMPIng.
//
// Copyright 2026 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief Example demonstrating the kamping-types module standalone (without the KaMPIng communicator).
///
/// Shows how to query type metadata at compile time, obtain MPI_Datatypes via mpi_type_traits,
/// manage custom types with ScopedDatatype, and use those types in raw MPI calls.

#include <array>
#include <iostream>
#include <utility>

#include <mpi.h>

#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/types/scoped_datatype.hpp"
#include "kamping/types/std/unsafe/trivially_copyable.hpp"
#include "kamping/types/std/utility.hpp"
#include "kamping/types/struct_type.hpp"

// -----------------------------------------------------------------------
// Manual mpi_type_traits specialization for a user-defined struct.
//
// Specialize mpi_type_traits in namespace kamping::types to register any
// type that struct_type can reflect (std::pair, std::tuple, or a
// Boost.PFR-reflectable aggregate when KAMPING_ENABLE_REFLECTION is set).
// -----------------------------------------------------------------------
struct Particle {
    float  position[3]; ///< x, y, z
    double mass;
};

namespace kamping::types {
template <>
struct mpi_type_traits<Particle> : struct_type<Particle> {};
} // namespace kamping::types

static_assert(kamping::types::has_static_type_v<Particle>);

// A plain trivially-copyable struct — registered automatically by trivially_copyable.hpp.
struct Point3D {
    float x, y, z;
};
static_assert(kamping::types::has_static_type_v<Point3D>);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // -----------------------------------------------------------------------
    // Section 1: Compile-time type metadata
    // -----------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "=== Compile-time type queries ===\n" << std::boolalpha;

        auto info = [](char const* name, bool supported, bool needs_commit) {
            std::cout << "  " << name << ": supported=" << supported << ", needs_commit=" << needs_commit << "\n";
        };

        // Builtins and enums are supported; no commit needed.
        info("int", kamping::types::has_static_type_v<int>, kamping::types::mpi_type_traits<int>::has_to_be_committed);
        info(
            "double",
            kamping::types::has_static_type_v<double>,
            kamping::types::mpi_type_traits<double>::has_to_be_committed
        );

        // C-array and std::array map to contiguous types; commit is required.
        info(
            "float[4]",
            kamping::types::has_static_type_v<float[4]>,
            kamping::types::mpi_type_traits<float[4]>::has_to_be_committed
        );
        info(
            "std::array<double,3>",
            kamping::types::has_static_type_v<std::array<double, 3>>,
            kamping::types::mpi_type_traits<std::array<double, 3>>::has_to_be_committed
        );

        // std::pair is handled via the struct_type specialization added above.
        info(
            "std::pair<int,double>",
            kamping::types::has_static_type_v<std::pair<int, double>>,
            kamping::types::mpi_type_traits<std::pair<int, double>>::has_to_be_committed
        );
    }

    // -----------------------------------------------------------------------
    // Section 2: Builtin type — no commit needed, use directly
    // -----------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n=== Builtin type: int -> MPI_INT ===\n";
        MPI_Datatype int_type = kamping::types::mpi_type_traits<int>::data_type();
        std::cout << "  mpi_type_traits<int>::data_type() == MPI_INT: " << std::boolalpha << (int_type == MPI_INT)
                  << ".\n";
        // Builtin types are predefined constants — no commit or free required.
    }

    // -----------------------------------------------------------------------
    // Section 3: Array type — commit via ScopedDatatype
    // -----------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n=== Contiguous type: float[4] -> MPI_Type_contiguous ===\n";
    }
    {
        // ScopedDatatype commits on construction and frees on destruction.
        kamping::types::ScopedDatatype arr_type{kamping::types::mpi_type_traits<float[4]>::data_type()};

        if (size >= 2) {
            float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
            if (rank == 0) {
                MPI_Send(data, 1, arr_type.data_type(), 1, 0, MPI_COMM_WORLD);
                std::cout << "  Rank 0 sent float[4] {1, 2, 3, 4}.\n";
            } else if (rank == 1) {
                float recv[4] = {};
                MPI_Recv(recv, 1, arr_type.data_type(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "  Rank 1 received: {" << recv[0] << ", " << recv[1] << ", " << recv[2] << ", " << recv[3]
                          << "}.\n";
            }
        } else if (rank == 0) {
            std::cout << "  (Run with at least 2 ranks to see the send/receive.)\n";
        }
    } // arr_type freed here

    // -----------------------------------------------------------------------
    // Section 4: Struct type — std::pair<int, double>
    // -----------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n=== Struct type: std::pair<int,double> -> MPI_Type_create_struct ===\n";
    }
    {
        kamping::types::ScopedDatatype pair_type{kamping::types::mpi_type_traits<std::pair<int, double>>::data_type()};

        if (size >= 2) {
            std::pair<int, double> value = {42, 3.14159};
            if (rank == 0) {
                MPI_Send(&value, 1, pair_type.data_type(), 1, 0, MPI_COMM_WORLD);
                std::cout << "  Rank 0 sent pair{42, 3.14159}.\n";
            } else if (rank == 1) {
                std::pair<int, double> recv{};
                MPI_Recv(&recv, 1, pair_type.data_type(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "  Rank 1 received: {" << recv.first << ", " << recv.second << "}.\n";
            }
        } else if (rank == 0) {
            std::cout << "  (Run with at least 2 ranks to see the send/receive.)\n";
        }
    } // pair_type freed here

    MPI_Finalize();
    return 0;
}
