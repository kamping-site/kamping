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

#include <algorithm>
#include <functional>
#include <type_traits>

namespace kamping {
namespace ops {
template <typename T>
struct max {
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::max(lhs, rhs);
    }
};

template <typename T>
struct min {
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::min(lhs, rhs);
    }
};

template <typename T>
using plus = std::plus<T>;

template <typename T>
using multiplies = std::multiplies<T>;

template <typename T>
using logical_and = std::logical_and<T>;

template <typename T>
using bit_and = std::bit_and<T>;

template <typename T>
using logical_or = std::logical_or<T>;

template <typename T>
using bit_or = std::bit_or<T>;

template <typename T>
struct logical_xor {
    constexpr bool operator()(const T& lhs, const T& rhs) const {
        return (lhs && !rhs) || (!lhs && rhs);
    }
};

template <typename T>
using bit_xor = std::bit_xor<T>;
} // namespace ops
namespace internal {
template <typename Op, typename T>
struct is_builtin_mpi_op : std::false_type {};


template <typename T>
struct is_builtin_mpi_op<kamping::ops::max<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_SUM;
    }
};

template <typename T>
struct is_builtin_mpi_op<std::plus<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_SUM;
    }
};

template <typename T>
struct is_builtin_mpi_op<std::plus<>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_SUM;
    }
};

template <typename T>
struct is_builtin_mpi_op<std::multiplies<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_PROD;
    }
};

template <typename T>
struct is_builtin_mpi_op<std::logical_and<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_LAND;
    }
};

template <typename T>
struct is_builtin_mpi_op<std::bit_and<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_BAND;
    }
};

template <typename T>
struct is_builtin_mpi_op<std::logical_or<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_LOR;
    }
};

template <typename T>
struct is_builtin_mpi_op<std::bit_or<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_BOR;
    }
};

template <typename T>
struct is_builtin_mpi_op<kamping::ops::logical_xor<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_LXOR;
    }
};

template <typename T>
struct is_builtin_mpi_op<std::bit_xor<T>, T> : std::true_type {
    static MPI_Op op() {
        return MPI_BXOR;
    }
};

///@todo support for MPI_MAXLOC and MPI_MINLOC

template <int is_commutative, typename Op, typename T>
struct UserOperation {
    UserOperation() {
        MPI_Op_create(UserOperation<is_commutative, Op, T>::execute, is_commutative, &mpi_op);
    }
    static void execute(void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
        T* invec_    = static_cast<T*>(invec);
        T* inoutvec_ = static_cast<T*>(inoutvec);
        Op op{};
        std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, op);
    }
    ~UserOperation() {
        MPI_Op_free(&mpi_op);
    }
    MPI_Op& get_mpi_op() {
        return mpi_op;
    }
    MPI_Op mpi_op;
    Op     op;
};

} // namespace internal
} // namespace kamping
