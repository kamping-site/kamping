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
/// @brief Definitions for builtin MPI operations

#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>

#include <mpi.h>

#include "kamping/mpi_datatype.hpp"

namespace kamping {
namespace internal {

/// @brief Wrapper struct for std::max
///
/// Other than the operators defined in `<functional>` like \c std::plus, \c std::max is a function and not a function
/// object. To enable template matching for detection of builtin MPI operations we therefore need to wrap it.
/// The actual implementation is used in case that the operation is a builtin operation for the given datatype.
///
/// @tparam T the type of the operands
template <typename T>
struct max_impl {
    /// @brief Returns the maximum of the two parameters
    /// @param lhs the first operand
    /// @param rhs the second operand
    /// @return the maximum
    constexpr T operator()(const T& lhs, const T& rhs) const {
        // return std::max<const T&>(lhs, rhs);
        return std::max(lhs, rhs);
    }
};

/// @brief Template specialization for \c kamping::internal::max_impl without type parameter, which leaves the operand
/// type to be deduced.
///
/// The actual implementation is used in case that the operation is a builtin operation for the given datatype.
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
/// The actual implementation is used in case that the operation is a builtin operation for the given datatype.
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
/// The actual implementation is used in case that the operation is a builtin operation for the given datatype.
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
///
/// The actual implementation is used in case that the operation is a builtin operation for the given datatype.
///
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
/// The actual implementation is used in case that the operation is a builtin operation for the given datatype.
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

/// @brief this namespace contains all builtin operations supported by MPI.
///
/// You can either use them by passing their STL counterparts like \c std::plus<>, \c std::multiplies<> etc. or using
/// the aliases \c kamping::ops::plus, \c kamping::ops::multiplies, \c kamping::ops::max(), ...
/// You can either use them without a template parameter (\c std::plus<>) or explicitly specify the type (\c
/// std::plus<int>). In the latter case, the type must match the datatype of the buffer the operation shall be applied
/// to.
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

namespace internal {
/// @brief tag for a commutative reduce operation
struct commutative_tag {};
/// @brief tag for a non-commutative reduce operation
struct non_commutative_tag {};
/// @brief tag for a reduce operation without manually declared commutativity (this is only used
/// internally for builtin reduce operations)
struct undefined_commutative_tag {};
} // namespace internal

[[maybe_unused]] constexpr internal::commutative_tag     commutative{};     ///< global tag for commutativity
[[maybe_unused]] constexpr internal::non_commutative_tag non_commutative{}; ///< global tag for non-commutativity

namespace internal {

#ifdef KAMPING_DOXYGEN_ONLY
/// @brief Type trait for checking whether a functor is a builtin MPI reduction operation and query corresponding \c
/// MPI_Op.
///
/// Example:
/// @code
/// is_builtin_mpi_op<kamping::ops::plus<>, int>::value // true
/// is_builtin_mpi_op<kamping::ops::plus<>, int>::op()  // MPI_SUM
/// is_builtin_mpi_op<std::plus<>, int>::value          // true
/// is_builtin_mpi_op<std::plus<>, int>::op()           // MPI_SUM
/// is_builtin_mpi_op<std::minus<>, int>::value         // false
/// //is_builtin_mpi_op<std::minus<>, int>::op()        // error: fails to compile because op is not defined
/// @endcode
///
/// @tparam Op type of the operation
/// @tparam Datatype type to apply the operation to
template <typename Op, typename Datatype>
struct mpi_operation_traits {
    /// @brief \c true if the operation defined by \c Op is a builtin MPI operation for the type \c Datatype
    ///
    /// Note that this is only true if the \c MPI_Datatype corresponding to the C++ datatype \c Datatype supports the
    /// operation according to the standard. If MPI supports the operation for this type, then this is true for functors
    /// defined in \c kamping::ops and there corresponding type-aliased equivalents in the standard library.
    static constexpr bool is_builtin;

    /// @brief The identity of this operation applied on this datatype.
    ///
    /// The identity of a {value, operation} pair is the value for which the following two equaltion hold:
    /// - `identity operation value = value`
    /// - `value operation identity = value`
    static constexpr T identity;

    /// @brief get the MPI_Op for a builtin type
    ///
    /// This member is only defined if \c value is \c true. It can then be used to query the predefined constant of
    /// type \c MPI_OP matching the functor defined by type \c Op, e.g. returns \c MPI_SUM if \c Op is \c
    /// kamping::ops::plus<>.
    /// @returns the builtin \c MPI_Op constant
    static MPI_Op op();
};
#else

template <typename Op, typename T, typename Enable = void>
struct mpi_operation_traits {
    static constexpr bool is_builtin = false;
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::max<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer
        || mpi_type_traits<T>::category == TypeCategory::floating)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = std::numeric_limits<T>::lowest();
    static MPI_Op         op() {
        return MPI_MAX;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::min<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer
        || mpi_type_traits<T>::category == TypeCategory::floating)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = std::numeric_limits<T>::max();
    static MPI_Op         op() {
        return MPI_MIN;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::plus<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer || mpi_type_traits<T>::category == TypeCategory::floating
        || mpi_type_traits<T>::category == TypeCategory::complex)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = 0;
    static MPI_Op         op() {
        return MPI_SUM;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::multiplies<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer || mpi_type_traits<T>::category == TypeCategory::floating
        || mpi_type_traits<T>::category == TypeCategory::complex)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = 1;
    static MPI_Op         op() {
        return MPI_PROD;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::logical_and<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer
        || mpi_type_traits<T>::category == TypeCategory::logical)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = true;
    static MPI_Op         op() {
        return MPI_LAND;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::logical_or<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer
        || mpi_type_traits<T>::category == TypeCategory::logical)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = false;
    static MPI_Op         op() {
        return MPI_LOR;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::logical_xor<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer
        || mpi_type_traits<T>::category == TypeCategory::logical)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = false;
    static MPI_Op         op() {
        return MPI_LXOR;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::bit_and<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer
        || mpi_type_traits<T>::category == TypeCategory::byte)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = ~(T{0});
    static MPI_Op         op() {
        return MPI_BAND;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::bit_or<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer
        || mpi_type_traits<T>::category == TypeCategory::byte)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = T{0};
    static MPI_Op         op() {
        return MPI_BOR;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::bit_xor<S>, T,
    typename std::enable_if<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        mpi_type_traits<T>::category == TypeCategory::integer
        || mpi_type_traits<T>::category == TypeCategory::byte)>::type> {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = 0;
    static MPI_Op         op() {
        return MPI_BXOR;
    }
};
#endif

/// @todo support for MPI_MAXLOC and MPI_MINLOC

/// @brief type used by user-defined operations passed to \c MPI_Op_create
using mpi_custom_operation_type = void (*)(void*, void*, int*, MPI_Datatype*);

/// @brief Wrapper for a user defined reduction operation based on a functor object.
///
/// Internally, this creates an \c MPI_Op which is freed upon destruction.
/// @tparam is_commutative whether the operation is commutative or not
/// @tparam T the type to apply the operation to.
/// @tparam Op type of the functor object to wrap
template <bool is_commutative, typename T, typename Op>
class UserOperationWrapper {
public:
    static_assert(
        std::is_default_constructible_v<Op>,
        "This wrapper only works with default constructible functors, i.e., not with lambdas.");
    void operator=(UserOperationWrapper<is_commutative, T, Op>&) = delete;
    void operator=(UserOperationWrapper<is_commutative, T, Op>&&) = delete;
    /// @brief creates an MPI operation for the specified functor
    /// @param op the functor to call for reduction.
    ///  this has to be a binary function applicable to two arguments of type \c T which return a result of type  \c T
    UserOperationWrapper(Op&& op [[maybe_unused]]) {
        static_assert(std::is_invocable_r_v<T, Op, T&, T&>, "Type of custom operation does not match.");
        MPI_Op_create(UserOperationWrapper<is_commutative, T, Op>::execute, is_commutative, &_mpi_op);
    }

    /// @brief wrapper around the provided functor which is called by MPI
    static void execute(void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
        T* invec_    = static_cast<T*>(invec);
        T* inoutvec_ = static_cast<T*>(inoutvec);
        Op op{};
        std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, op);
    }

    ~UserOperationWrapper() {
        MPI_Op_free(&_mpi_op);
    }

    /// @returns the \c MPI_Op constructed for the provided functor.
    ///
    /// Do not free this operation manually, because the destructor calls it. Some MPI implementations silently segfault
    /// if an \c MPI_Op is freed multiple times.
    MPI_Op get_mpi_op() {
        return _mpi_op;
    }

private:
    MPI_Op _mpi_op; ///< the \c MPI_Op referencing the user defined operation
};

/// @brief Wrapper for a user defined reduction operation based on a function pointer.
///
/// Internally, this creates an \c MPI_Op which is freed upon destruction.
/// @tparam is_commutative whether the operation is commutative or not
template <bool is_commutative>
class UserOperationPtrWrapper {
public:
    UserOperationPtrWrapper<is_commutative>& operator=(UserOperationPtrWrapper<is_commutative> const&) = delete;

    /// @brief move assignment
    UserOperationPtrWrapper<is_commutative>& operator=(UserOperationPtrWrapper<is_commutative>&& other_op) {
        this->_mpi_op   = other_op._mpi_op;
        this->_no_op    = other_op._no_op;
        other_op._no_op = true;
        return *this;
    }

    UserOperationPtrWrapper<is_commutative>(UserOperationPtrWrapper<is_commutative> const&) = delete;
    /// @brief move constructor
    UserOperationPtrWrapper<is_commutative>(UserOperationPtrWrapper<is_commutative>&& other_op) {
        this->_mpi_op   = other_op._mpi_op;
        this->_no_op    = other_op._no_op;
        other_op._no_op = true;
    }
    /// @brief creates an empty operation wrapper
    UserOperationPtrWrapper() : _no_op(true) {
        _mpi_op = MPI_OP_NULL;
    }
    /// @brief creates an MPI operation for the specified function pointer
    /// @param ptr the functor to call for reduction
    /// this parameter must match the semantics of the function pointer passed to \c MPI_Op_create according to the
    /// MPI standard.
    UserOperationPtrWrapper(mpi_custom_operation_type ptr) : _no_op(false) {
        KASSERT(ptr != nullptr);
        MPI_Op_create(ptr, is_commutative, &_mpi_op);
    }

    ~UserOperationPtrWrapper() {
        if (!_no_op) {
            MPI_Op_free(&_mpi_op);
        }
    }

    /// @returns the \c MPI_Op constructed for the provided functor.
    ///
    /// Do not free this operation manually, because the destructor calls it. Some MPI implementations silently
    /// segfault if an \c MPI_Op is freed multiple times.
    MPI_Op get_mpi_op() {
        return _mpi_op;
    }

private:
    bool _no_op;    ///< indicates if this operation is empty or was moved, so we can avoid freeing the same operation
                    ///< multiple times upon destruction
    MPI_Op _mpi_op; ///< the \c MPI_Op referencing the user defined operation
};

#ifdef KAMPING_DOXYGEN_ONLY

/// @brief Wraps an operation and translates it to a builtin \c MPI_Op or constructs a custom operation.
/// @tparam T the argument type of the operation
/// @tparam Op the type of the operation
/// @tparam Commutative tag indicating if this type is commutative
template <typename T, typename Op, typename Commutative>
class ReduceOperation {
public:
    /// @brief Constructs on operation wrapper
    /// @param op the operation
    /// maybe a function object a lambda or a \c std::function
    /// @param commutative
    /// May be any instance of \c commutative, \c or non_commutative. Passing \c undefined_commutative is only
    /// supported for builtin operations.
    ReduceOperation(Op&& op, Commutative commutative);

    static constexpr bool is_builtin;  ///< indicates if this is a builtin operation
    static constexpr bool commutative; ///< indicates if this operation is commutative

    ///  @returns the \c MPI_Op associated with this operation
    MPI_Op op();
};

#else

template <typename T, typename Op, typename Commutative, class Enable = void>
class ReduceOperation {
    static_assert(
        std::is_same_v<
            Commutative,
            kamping::internal::commutative_tag> || std::is_same_v<Commutative, kamping::internal::non_commutative_tag>,
        "For custom operations you have to specify whether they are commutative.");

public:
    ReduceOperation(Op&& op, Commutative) : _operation(std::move(op)) {}
    static constexpr bool is_builtin  = false;
    static constexpr bool commutative = std::is_same_v<Commutative, kamping::internal::commutative_tag>;
    MPI_Op                op() {
        return _operation.get_mpi_op();
    }

private:
    UserOperationWrapper<commutative, T, Op> _operation;
};

template <typename T, typename Op, typename Commutative>
class ReduceOperation<T, Op, Commutative, typename std::enable_if<mpi_operation_traits<Op, T>::is_builtin>::type> {
    static_assert(
        std::is_same_v<Commutative, kamping::internal::undefined_commutative_tag>,
        "For builtin operations you don't need to specify whether they are commutative.");

public:
    ReduceOperation(Op&&, Commutative) {}
    static constexpr bool is_builtin  = true;
    static constexpr bool commutative = true; // builtin operations are always commutative
    MPI_Op                op() {
        return mpi_operation_traits<Op, T>::op();
    }
};

template <typename T, typename Op, typename Commutative>
class ReduceOperation<T, Op, Commutative, typename std::enable_if<!std::is_default_constructible_v<Op> >::type> {
    static_assert(
        std::is_same_v<
            Commutative,
            kamping::internal::commutative_tag> || std::is_same_v<Commutative, kamping::internal::non_commutative_tag>,
        "For custom operations you have to specify whether they are commutative.");

public:
    ReduceOperation(Op&& op, Commutative) : _operation() {
        // A lambda is may not be default constructed nor copied, so we need some hacks to deal with them.
        // Because each lambda has a distinct type we initiate the static Op here and can access it from the static
        // context of function pointer created afterwards.
        static Op func = op;

        mpi_custom_operation_type ptr = [](void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
            T* invec_    = static_cast<T*>(invec);
            T* inoutvec_ = static_cast<T*>(inoutvec);
            std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, func);
        };
        _operation = {ptr};
    }
    static constexpr bool is_builtin  = false;
    static constexpr bool commutative = std::is_same_v<Commutative, kamping::internal::commutative_tag>;
    MPI_Op                op() {
        return _operation.get_mpi_op();
    }

private:
    UserOperationPtrWrapper<commutative> _operation;
};
#endif
} // namespace internal
} // namespace kamping
