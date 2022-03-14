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

#pragma once 

#include <mpi.h>

#include <algorithm>
#include <type_traits>

/// @brief obsolete by Niklas PR
template <int is_commutative, typename Op, typename T>
struct CustomFunction {
    /// @brief obsolete by Niklas PR
    CustomFunction() {
        MPI_Op_create(CustomFunction<is_commutative, Op, T>::execute, is_commutative, &mpi_op);
    }

    /// @brief obsolete by Niklas PR
    static void execute(void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
        T* invec_    = static_cast<T*>(invec);
        T* inoutvec_ = static_cast<T*>(inoutvec);
        Op op{};
        std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, op);
    }

    /// @brief obsolete by Niklas PR
    ~CustomFunction() {
        MPI_Op_free(&mpi_op);
    }

    /// @brief obsolete by Niklas PR
    MPI_Op& get_mpi_op() {
        return mpi_op;
    }

    /// @brief obsolete by Niklas PR
    MPI_Op mpi_op;

    /// @brief obsolete by Niklas PR
    Op op;
};
