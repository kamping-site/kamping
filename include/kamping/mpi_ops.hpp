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

/// @file
/// @brief Definitions for builtin MPI operations

#pragma once

#include <mpi.h>

#include <algorithm>
#include <functional>
#include <type_traits>

#include "kamping/mpi_datatype.hpp"


namespace kamping {
namespace internal {

/// @brief Wrapper struct for std::max
///
/// Other than the operators defined in `<functional>` like \c std::plus, \c std::max is a function and not a function
/// object. To enable template matching for detection of builtin MPI operations we therefore need to wrap it.
///
/// @tparam T the type of the operands
template <typename T>
struct max_impl {
    /// @brief Returns the maximum of the two parameters
    /// @param lhs the first operand
    /// @param rhs the second operand
    /// @return the maximum
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::max(lhs, rhs);
    }
};

/// @brief Template specialization for \c kamping::internal::max_impl without type parameter, which leaves the operand
/// type to be deduced.
template <>
struct max_impl<void> {
    /// @brief Returns the maximum of the two parameters
    /// @param lhs the first operand
    /// @param rhs the second operand
    /// @tparam T the type of the operands
    /// @return the maximum
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::max(lhs, rhs);
    }
};

/// @brief Wrapper struct for std::min
///
/// Other than the operators defined in `<functional>` like \c std::plus, \c std::min is a function and not a function
/// object. To enable template matching for detection of builtin MPI operations we therefore need to wrap it.
///
/// @tparam T the type of the operands
template <typename T>
struct min_impl {
    /// @brief returns the maximum of the two parameters
    /// @param lhs the first operand
    /// @param rhs the second operand
    /// @return the maximum
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::min(lhs, rhs);
    }
};

/// @brief Template specialization for \c kamping::internal::min_impl without type parameter, which leaves the operand
/// type to be deduced.
template <>
struct min_impl<void> {
    /// @brief Returns the maximum of the two parameters
    /// @param lhs the first operand
    /// @param rhs the second operand
    /// @tparam T the type of the operands
    /// @return the maximum
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const {
        return std::min(lhs, rhs);
    }
};

/// @brief Wrapper struct for logical xor, as the standard library does not provided a function object for it.
/// @tparam T type of the operands
template <typename T>
struct logical_xor_impl {
    /// @brief Returns the logical xor of the two parameters
    /// @param lhs the first operand
    /// @param rhs the second operand
    /// @return the logical xor
    constexpr bool operator()(const T& lhs, const T& rhs) const {
        return (lhs && !rhs) || (!lhs && rhs);
    }
};

/// @brief Template specialization for \c kamping::internal::logical_xor_impl without type parameter, which leaves to
/// operand type to be deduced.
template <>
struct logical_xor_impl<void> {
    /// @brief Returns the logical xor of the two parameters
    /// @param lhs the left operand
    /// @param rhs the right operand
    /// @tparam T type of the left operand
    /// @tparam S type of the right operand
    /// @return the logical xor
    template <typename T, typename S>
    constexpr bool operator()(const T& lhs, const S& rhs) const {
        return (lhs && !rhs) || (!lhs && rhs);
    }
};
} // namespace internal

namespace ops {


/// @brief builtin maximum operation (aka `MPI_MAX`)
template <typename T = void>
using max = kamping::internal::max_impl<T>;

/// @brief builtin minimum operation (aka `MPI_MIN`)
template <typename T = void>
using min = kamping::internal::min_impl<T>;

/// @brief builtin summation operation (aka `MPI_SUM`)
template <typename T = void>
using plus = std::plus<T>;

/// @brief builtin multiplication operation (aka `MPI_PROD`)
template <typename T = void>
using multiplies = std::multiplies<T>;

/// @brief builtin logical and operation (aka `MPI_LAND`)
template <typename T = void>
using logical_and = std::logical_and<T>;

/// @brief builtin bitwise and operation (aka `MPI_BAND`)
template <typename T = void>
using bit_and = std::bit_and<T>;

/// @brief builtin logical or operation (aka `MPI_LOR`)
template <typename T = void>
using logical_or = std::logical_or<T>;

/// @brief builtin bitwise or operation (aka `MPI_BOR`)
template <typename T = void>
using bit_or = std::bit_or<T>;

/// @brief builtin logical xor operation (aka `MPI_LXOR`)
template <typename T = void>
using logical_xor = kamping::internal::logical_xor_impl<T>;

/// @brief builtin bitwise xor operation (aka `MPI_BXOR`)
template <typename T = void>
using bit_xor = std::bit_xor<T>;

} // namespace ops

/// @brief tag for a commutative reduce operation
struct commutative {};
/// @brief tag for a non-commutative reduce operation
struct non_commutative {};
/// @brief tag for a reduce operation without manually declared commutativity (this is only used
/// internally for builtin reduce operations)
struct undefined_commutative {};

namespace internal {


#ifdef KAMPING_DOXYGEN_ONLY
template <typename Op, typename T>
struct is_builtin_mpi_op {
    constexpr bool value;
    static MPI_Op op();
};
#else

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
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_float || mpi_type_traits<T>::is_complex)>::type>
    : std::true_type {
    static MPI_Op op() {
        return MPI_SUM;
    }
};

template <typename T, typename S>
struct is_builtin_mpi_op<
    kamping::ops::multiplies<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::is_integer || mpi_type_traits<T>::is_float || mpi_type_traits<T>::is_complex)>::type>
    : std::true_type {
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
#endif

///@todo support for MPI_MAXLOC and MPI_MINLOC

template <int is_commutative, typename Op, typename T>
struct UserOperation {
    void operator=(UserOperation<is_commutative, Op, T>&) = delete;
    void operator=(UserOperation<is_commutative, Op, T>&&) = delete;
    UserOperation(Op&& op [[maybe_unused]]) {
        MPI_Op_create(UserOperation<is_commutative, Op, T>::execute, is_commutative, &mpi_op);
    }

    /// @brief obsolete by Niklas PR
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
    // static func_type func_op;
    MPI_Op mpi_op;
};

using mpi_custom_operation_type = void (*)(void*, void*, int*, MPI_Datatype*);

template <int is_commutative>
struct UserOperationPtr {
    void operator=(UserOperationPtr<is_commutative>&) = delete;
    void operator                                     =(UserOperationPtr<is_commutative>&& other_op) {
        this->mpi_op   = other_op.mpi_op;
        this->no_op    = other_op.no_op;
        other_op.no_op = true;
    }
    UserOperationPtr() : no_op(true) {
        mpi_op = MPI_OP_NULL;
    }
    UserOperationPtr(mpi_custom_operation_type ptr) : no_op(false) {
        KASSERT(ptr != nullptr);
        MPI_Op_create(ptr, is_commutative, &mpi_op);
    }

    ~UserOperationPtr() {
        if (!no_op) {
            MPI_Op_free(&mpi_op);
        }
    }

    /// @brief obsolete by Niklas PR
    MPI_Op& get_mpi_op() {
        return mpi_op;
    }

    /// @brief obsolete by Niklas PR
    bool   no_op;
    MPI_Op mpi_op;
};

} // namespace internal
} // namespace kamping
