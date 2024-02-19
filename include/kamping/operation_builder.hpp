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
/// The class defined in this file serve as wrapper for functions passed to \c MPI calls wrapped by KaMPIng.

#pragma once

#include "kamping/assertion_levels.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kassert/kassert.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

namespace internal {

/// @brief Parameter wrapping an operation passed to reduce-like MPI collectives.
/// This wraps an MPI operation without the argument of the operation specified. This enables the user to construct
/// such wrapper using the parameter factory \c kamping::op without passing the type of the operation. The library
/// developer may then construct the actual operation wrapper with a given type later.
///
/// @tparam Op type of the operation (may be a function object or a lambda)
/// @tparam Commutative tag specifying if the operation is commutative
template <typename Op, typename Commutative>
class OperationBuilder {
public:
    static constexpr ParameterType parameter_type =
        ParameterType::op; ///< The type of parameter this object encapsulates.

    /// @brief constructs an Operation builder
    /// @param op the operation
    /// @param commutative_tag tag indicating if the operation is commutative (see \c kamping::op for details)
    OperationBuilder(Op&& op, Commutative commutative_tag [[maybe_unused]]) : _op(op) {}

    /// @brief Move constructor for OperationsBuilder.
    OperationBuilder(OperationBuilder&&) = default;

    /// @brief Move assignment operator for OperationsBuilder.
    OperationBuilder& operator=(OperationBuilder&&) = default;

    /// @brief Copy constructor is deleted as buffers should only be moved.
    OperationBuilder(OperationBuilder const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    OperationBuilder& operator=(OperationBuilder const&) = delete;
    // redundant as defaulted move constructor implies the deletion

    /// @brief constructs an operation for the given type T
    /// @tparam T argument type of the reduction operation
    template <typename T>
    [[nodiscard]] auto build_operation() {
        if constexpr (std::is_same_v<std::remove_reference_t<std::remove_const_t<Op>>, MPI_Op>) {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
            // mapping a MPI_Op to the corresponding function object requires a scan over all builtin operations
            with_operation_functor(_op, [](auto operation) {
                if constexpr (!std::is_same_v<decltype(operation), ops::null<>>) {
                    // the user passed a builtin datatype, so we can do some checking
                    KASSERT(
                        (mpi_operation_traits<decltype(operation), T>::is_builtin),
                        "The provided builtin operation is not compatible with datatype T."
                    );
                }
            });
#endif
            return ReduceOperation<T, MPI_Op, ops::internal::undefined_commutative_tag>(_op);
        } else {
            static_assert(std::is_invocable_r_v<T, Op, T const&, T const&>, "Type of custom operation does not match.");
            return ReduceOperation<T, Op, Commutative>(std::forward<Op>(_op), Commutative{});
        }
    }

private:
    Op _op; ///< the operation which is encapsulated
};

} // namespace internal

/// @}

} // namespace kamping
