// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <mpi.h>

#include <algorithm>
#include <type_traits>

template <int is_commutative, typename Op, typename T>
struct CustomFunction {
    CustomFunction() {
        MPI_Op_create(CustomFunction<is_commutative, Op, T>::execute, is_commutative, &mpi_op);
    }
    static void execute(void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
        T* invec_    = static_cast<T*>(invec);
        T* inoutvec_ = static_cast<T*>(inoutvec);
        Op op{};
        std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, op);
    }
    ~CustomFunction() {
        MPI_Op_free(&mpi_op);
    }
    MPI_Op& get_mpi_op() {
        return mpi_op;
    }
    MPI_Op mpi_op;
    Op     op;
};
