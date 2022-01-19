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
#include <functional>
#include <type_traits>

#include "kamping/mpi_datatype.hpp"

namespace kamping {
namespace internal {
template <typename T>
struct max_impl {
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::max(lhs, rhs);
    }
};

template <>
struct max_impl<void> {
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::max(lhs, rhs);
    }
};

template <typename T>
struct min_impl {
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::min(lhs, rhs);
    }
};

template <>
struct min_impl<void> {
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::min(lhs, rhs);
    }
};
template <typename T>
struct logical_xor_impl {
    constexpr bool operator()(const T& lhs, const T& rhs) const {
        return (lhs && !rhs) || (!lhs && rhs);
    }
};

template <>
struct logical_xor_impl<void> {
    template <typename T, typename S>
    constexpr bool operator()(const T& lhs, const S& rhs) const {
        return (lhs && !rhs) || (!lhs && rhs);
    }
};
} // namespace internal
namespace ops {


template <typename T = void>
using max = kamping::internal::max_impl<T>;

template <typename T = void>
using min = kamping::internal::min_impl<T>;

template <typename T = void>
using plus = std::plus<T>;

template <typename T = void>
using multiplies = std::multiplies<T>;

template <typename T = void>
using logical_and = std::logical_and<T>;

template <typename T = void>
using bit_and = std::bit_and<T>;

template <typename T = void>
using logical_or = std::logical_or<T>;

template <typename T = void>
using bit_or = std::bit_or<T>;

template <typename T = void>
using logical_xor = kamping::internal::logical_xor_impl<T>;

template <typename T = void>
using bit_xor = std::bit_xor<T>;

} // namespace ops

namespace internal {

template <typename Op, typename T, typename Enable = void>
struct is_builtin_mpi_op : std::false_type {};


template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::max<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_float)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_MAX;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::min<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_float)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_MIN;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::plus<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_float || mpi_type_traits<T>::is_complex)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_SUM;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::multiplies<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_float || mpi_type_traits<T>::is_complex)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_PROD;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::logical_and<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_logical)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_LAND;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::logical_or<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_logical)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_LOR;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::logical_xor<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_logical)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_LXOR;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::bit_and<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_byte)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_BAND;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::bit_or<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_byte)>::type> : std::true_type {
    static MPI_Op op() {
        return MPI_BOR;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::bit_xor<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_byte)>::type> : std::true_type {
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
