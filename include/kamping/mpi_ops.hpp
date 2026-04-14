// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
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
/// @brief MPI reduction operation wrappers.
///
/// The functor vocabulary (`kamping::ops::`), type traits (`mpi_operation_traits`),
/// `ScopedOp`, `ScopedFunctorOp`, and `ScopedCallbackOp` live in
/// `kamping/types/reduce_ops.hpp` (the `kamping-types` module) and are included from there.
/// This file adds `ReduceOperation`, which selects among these building blocks based on
/// the operation and commutative tag types.

#pragma once

#include <algorithm>
#include <type_traits>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/types/reduce_ops.hpp"

namespace kamping {
namespace internal {

// Bring kamping::types::mpi_operation_traits into kamping::internal so that existing code
// that references kamping::internal::mpi_operation_traits<Op, T> continues to compile without
// change. Specializations are resolved through the primary template in kamping::types.
using kamping::types::mpi_operation_traits;

// Bring with_operation_functor into kamping::internal (backward compat).
using kamping::types::with_operation_functor;

// ---------------------------------------------------------------------------
// ReduceOperation — high-level op wrapper used by collectives
// ---------------------------------------------------------------------------

#ifdef KAMPING_DOXYGEN_ONLY

/// @brief Wraps an operation and translates it to a builtin \c MPI_Op or constructs a custom operation.
/// @tparam T the argument type of the operation
/// @tparam Op the type of the operation
/// @tparam Commutative tag indicating if this type is commutative
template <typename T, typename Op, typename Commutative>
class ReduceOperation {
public:
    /// @brief Constructs an operation wrapper.
    /// @param op the operation (function object, lambda, or \c std::function)
    /// @param commutative commutativity tag (\c kamping::ops::commutative or \c kamping::ops::non_commutative)
    ReduceOperation(Op&& op, Commutative commutative);

    static constexpr bool is_builtin;  ///< True if this is a predefined MPI operation.
    static constexpr bool commutative; ///< True if the operation is commutative.

    /// @returns the \c MPI_Op associated with this operation.
    MPI_Op op();

    /// @brief Call the underlying operation with the provided arguments.
    T operator()(T const& lhs, T const& rhs) const;

    /// @brief Returns the identity element for this operation and data type.
    ///
    /// Only available when `is_builtin == true`. For custom operations this member does not exist;
    /// callers must guard with `if constexpr (operation.is_builtin)`.
    T identity();
};

#else

// Primary: custom default-constructible functor.
template <typename T, typename Op, typename Commutative, class Enable = void>
class ReduceOperation {
    static_assert(
        std::is_same_v<
            Commutative,
            kamping::ops::internal::
                commutative_tag> || std::is_same_v<Commutative, kamping::ops::internal::non_commutative_tag>,
        "For custom operations you have to specify whether they are commutative."
    );

public:
    ReduceOperation(Op&& op, Commutative) : _operation(std::move(op)) {}
    static constexpr bool is_builtin  = false;
    static constexpr bool commutative = std::is_same_v<Commutative, kamping::ops::internal::commutative_tag>;

    T operator()(T const& lhs, T const& rhs) const {
        return _operation(lhs, rhs);
    }

    MPI_Op op() {
        return _operation.get();
    }

private:
    kamping::types::ScopedFunctorOp<commutative, T, Op> _operation;
};

// Specialization: raw MPI_Op passthrough.
template <typename T>
class ReduceOperation<T, MPI_Op, ops::internal::undefined_commutative_tag, void> {
public:
    ReduceOperation(MPI_Op op, ops::internal::undefined_commutative_tag = {}) : _op(op) {}
    static constexpr bool is_builtin = false;

    T operator()(T const& lhs, T const& rhs) const {
        KAMPING_ASSERT(_op != MPI_OP_NULL, "Cannot call MPI_OP_NULL.");
        T result;
        internal::with_operation_functor(_op, [&result, lhs, rhs, this](auto operation) {
            if constexpr (!std::is_same_v<decltype(operation), ops::null<> >) {
                result = operation(lhs, rhs);
            } else {
                result = rhs;
                MPI_Reduce_local(&lhs, &result, 1, mpi_datatype<T>(), _op);
            }
        });
        return result;
    }

    MPI_Op op() {
        return _op;
    }

private:
    MPI_Op _op;
};

// Specialization: builtin op — maps directly to a predefined MPI_Op constant.
template <typename T, typename Op, typename Commutative>
class ReduceOperation<T, Op, Commutative, std::enable_if_t<mpi_operation_traits<Op, T>::is_builtin> > {
    static_assert(
        std::is_same_v<Commutative, kamping::ops::internal::undefined_commutative_tag>,
        "For builtin operations you don't need to specify whether they are commutative."
    );

public:
    ReduceOperation(Op&&, Commutative) {}
    static constexpr bool is_builtin  = true;
    static constexpr bool commutative = true;

    MPI_Op op() {
        return mpi_operation_traits<Op, T>::op();
    }

    T operator()(T const& lhs, T const& rhs) const {
        return Op{}(lhs, rhs);
    }

    T identity() {
        return mpi_operation_traits<Op, T>::identity;
    }
};

// Specialization: non-default-constructible functor (lambda with captures).
template <typename T, typename Op, typename Commutative>
class ReduceOperation<T, Op, Commutative, std::enable_if_t<!std::is_default_constructible_v<Op> > > {
    static_assert(
        std::is_same_v<
            Commutative,
            kamping::ops::internal::
                commutative_tag> || std::is_same_v<Commutative, kamping::ops::internal::non_commutative_tag>,
        "For custom operations you have to specify whether they are commutative."
    );

public:
    ReduceOperation(Op&& op, Commutative) : _op(op) {
        // Each lambda type is distinct, so a static Op per instantiation is safe for a single
        // concurrent reduction.
        static Op func = _op;

        typename kamping::types::ScopedCallbackOp<commutative>::callback_type ptr =
            [](void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
                T* invec_    = static_cast<T*>(invec);
                T* inoutvec_ = static_cast<T*>(inoutvec);
                std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, func);
            };
        _operation = kamping::types::ScopedCallbackOp<commutative>{ptr};
    }
    static constexpr bool is_builtin  = false;
    static constexpr bool commutative = std::is_same_v<Commutative, kamping::ops::internal::commutative_tag>;

    MPI_Op op() {
        return _operation.get();
    }

    T operator()(T const& lhs, T const& rhs) const {
        return _op(lhs, rhs);
    }

private:
    Op                                            _op;
    kamping::types::ScopedCallbackOp<commutative> _operation;
};

#endif // KAMPING_DOXYGEN_ONLY

} // namespace internal
} // namespace kamping
