// This file is part of KaMPIng.
//
// Copyright 2021-2026 The KaMPIng Authors
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
/// @brief MPI reduction operation functor vocabulary, type traits, and RAII handle.
///
/// Provides:
/// - `kamping::ops::` — functor types and commutativity tags
/// - `kamping::types::mpi_operation_traits<Op, T>` — maps (functor, element type) → `MPI_Op`
/// - `kamping::types::ScopedOp` — RAII wrapper for an `MPI_Op`
/// - `kamping::types::ScopedFunctorOp` — creates an `MPI_Op` from a default-constructible C++ functor
/// - `kamping::types::ScopedCallbackOp` — creates an `MPI_Op` from a raw MPI callback function pointer
/// - `kamping::types::with_operation_functor` — maps a runtime `MPI_Op` to its functor

#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "kamping/types/builtin_types.hpp"

// ---------------------------------------------------------------------------
// kamping::ops — functor vocabulary
// ---------------------------------------------------------------------------

namespace kamping::ops::internal {

/// @brief Wrapper struct for std::max.
///
/// `std::max` is a function, not a function object. This wrapper allows template matching for
/// builtin MPI operation detection. The `<void>` specialization uses type deduction.
/// @tparam T the type of the operands
template <typename T>
struct max_impl {
    /// @brief Returns the maximum of the two parameters.
    /// @param lhs the first operand
    /// @param rhs the second operand
    constexpr T operator()(T const& lhs, T const& rhs) const {
        return std::max(lhs, rhs);
    }
};
/// @brief Template specialization of max_impl without type parameter, leaving the operand type to be deduced.
template <>
struct max_impl<void> {
    /// @brief Returns the maximum of the two parameters.
    /// @tparam T the type of the operands
    /// @param lhs the first operand
    /// @param rhs the second operand
    template <typename T>
    constexpr auto operator()(T const& lhs, T const& rhs) const {
        return std::max(lhs, rhs);
    }
};

/// @brief Wrapper struct for std::min (same rationale as max_impl).
/// @tparam T the type of the operands
template <typename T>
struct min_impl {
    /// @brief Returns the minimum of the two parameters.
    /// @param lhs the first operand
    /// @param rhs the second operand
    constexpr T operator()(T const& lhs, T const& rhs) const {
        return std::min(lhs, rhs);
    }
};
/// @brief Template specialization of min_impl without type parameter, leaving the operand type to be deduced.
template <>
struct min_impl<void> {
    /// @brief Returns the minimum of the two parameters.
    /// @tparam T the type of the operands
    /// @param lhs the first operand
    /// @param rhs the second operand
    template <typename T>
    constexpr auto operator()(T const& lhs, T const& rhs) const {
        return std::min(lhs, rhs);
    }
};

/// @brief Logical XOR function object (no STL equivalent).
/// @tparam T type of the operands
template <typename T>
struct logical_xor_impl {
    /// @brief Returns the logical XOR of the two parameters.
    /// @param lhs the first operand
    /// @param rhs the second operand
    constexpr bool operator()(T const& lhs, T const& rhs) const {
        return (lhs && !rhs) || (!lhs && rhs);
    }
};
/// @brief Template specialization of logical_xor_impl without type parameter, leaving operand types to be deduced.
template <>
struct logical_xor_impl<void> {
    /// @brief Returns the logical XOR of the two parameters.
    /// @tparam T type of the left operand
    /// @tparam S type of the right operand
    /// @param lhs the left operand
    /// @param rhs the right operand
    template <typename T, typename S>
    constexpr bool operator()(T const& lhs, S const& rhs) const {
        return (lhs && !rhs) || (!lhs && rhs);
    }
};

/// @brief Tag for a commutative user-defined reduce operation.
struct commutative_tag {};
/// @brief Tag for a non-commutative user-defined reduce operation.
struct non_commutative_tag {};
/// @brief Tag for a reduce operation without a manually declared commutativity (builtin ops only).
struct undefined_commutative_tag {};

} // namespace kamping::ops::internal

namespace kamping::ops {

/// @brief Builtin maximum operation (`MPI_MAX`).
template <typename T = void>
using max = kamping::ops::internal::max_impl<T>;

/// @brief Builtin minimum operation (`MPI_MIN`).
template <typename T = void>
using min = kamping::ops::internal::min_impl<T>;

/// @brief Builtin summation (`MPI_SUM`).
template <typename T = void>
using plus = std::plus<T>;

/// @brief Builtin multiplication (`MPI_PROD`).
template <typename T = void>
using multiplies = std::multiplies<T>;

/// @brief Builtin logical AND (`MPI_LAND`).
template <typename T = void>
using logical_and = std::logical_and<T>;

/// @brief Builtin bitwise AND (`MPI_BAND`).
template <typename T = void>
using bit_and = std::bit_and<T>;

/// @brief Builtin logical OR (`MPI_LOR`).
template <typename T = void>
using logical_or = std::logical_or<T>;

/// @brief Builtin bitwise OR (`MPI_BOR`).
template <typename T = void>
using bit_or = std::bit_or<T>;

/// @brief Builtin logical XOR (`MPI_LXOR`).
template <typename T = void>
using logical_xor = kamping::ops::internal::logical_xor_impl<T>;

/// @brief Builtin bitwise XOR (`MPI_BXOR`).
template <typename T = void>
using bit_xor = std::bit_xor<T>;

/// @brief Null operation (`MPI_OP_NULL`).
template <typename T = void>
struct null {};

[[maybe_unused]] constexpr internal::commutative_tag     commutative{};     ///< Tag: operation is commutative.
[[maybe_unused]] constexpr internal::non_commutative_tag non_commutative{}; ///< Tag: operation is non-commutative.

} // namespace kamping::ops

// ---------------------------------------------------------------------------
// kamping::types — mpi_operation_traits, ScopedOp, with_operation_functor
// ---------------------------------------------------------------------------

namespace kamping::types {

#ifdef KAMPING_DOXYGEN_ONLY
/// @brief Type trait that maps a (functor type, element type) pair to its builtin `MPI_Op`.
///
/// `mpi_operation_traits<Op, T>::is_builtin` is `true` when `Op` applied to `T` corresponds to
/// a predefined MPI operation constant. When `true`, `::op()` returns that constant and
/// `::identity` holds the identity element for the operation.
///
/// Example:
/// @code
/// mpi_operation_traits<kamping::ops::plus<>, int>::is_builtin  // true
/// mpi_operation_traits<kamping::ops::plus<>, int>::op()         // MPI_SUM
/// mpi_operation_traits<std::plus<>, int>::is_builtin            // true
/// mpi_operation_traits<std::minus<>, int>::is_builtin           // false
/// @endcode
/// @tparam Op     Functor type of the operation.
/// @tparam T      Element type to apply the operation to.
template <typename Op, typename T>
struct mpi_operation_traits {
    /// @brief \c true if \c Op applied to \c T corresponds to a predefined MPI operation constant.
    static constexpr bool is_builtin;

    /// @brief The identity element for this operation and data type.
    ///
    /// Only defined when \c is_builtin is \c true.
    static constexpr T identity;

    /// @brief Returns the predefined \c MPI_Op constant for this operation.
    ///
    /// Only defined when \c is_builtin is \c true.
    static MPI_Op op();
};
#else

template <typename Op, typename T, typename Enable = void>
struct mpi_operation_traits {
    static constexpr bool is_builtin = false;
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::max<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::floating
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = std::numeric_limits<T>::lowest();
    static MPI_Op         op() {
                return MPI_MAX;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::min<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::floating
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = std::numeric_limits<T>::max();
    static MPI_Op         op() {
                return MPI_MIN;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::plus<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::floating
        || builtin_type<T>::category == TypeCategory::complex
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = 0;
    static MPI_Op         op() {
                return MPI_SUM;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::multiplies<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::floating
        || builtin_type<T>::category == TypeCategory::complex
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = 1;
    static MPI_Op         op() {
                return MPI_PROD;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::logical_and<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::logical
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = true;
    static MPI_Op         op() {
                return MPI_LAND;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::logical_or<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::logical
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = false;
    static MPI_Op         op() {
                return MPI_LOR;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::logical_xor<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::logical
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = false;
    static MPI_Op         op() {
                return MPI_LXOR;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::bit_and<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::byte
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = ~(T{0});
    static MPI_Op         op() {
                return MPI_BAND;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::bit_or<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::byte
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = T{0};
    static MPI_Op         op() {
                return MPI_BOR;
    }
};

template <typename T, typename S>
struct mpi_operation_traits<
    kamping::ops::bit_xor<S>,
    T,
    std::enable_if_t<(std::is_same_v<S, void> || std::is_same_v<T, S>)&&(
        builtin_type<T>::category == TypeCategory::integer || builtin_type<T>::category == TypeCategory::byte
    )> > {
    static constexpr bool is_builtin = true;
    static constexpr T    identity   = T{0};
    static MPI_Op         op() {
                return MPI_BXOR;
    }
};

#endif // KAMPING_DOXYGEN_ONLY

// ---------------------------------------------------------------------------
// ScopedOp — RAII handle for MPI_Op
// ---------------------------------------------------------------------------

/// @brief RAII wrapper for an `MPI_Op`.
///
/// Calls `MPI_Op_free` on destruction only when `owns` is true (i.e. the op was created via
/// `MPI_Op_create` for a user-defined functor). Predefined MPI constants (`MPI_SUM`,
/// `MPI_MAX`, …) are never freed.
///
/// Analogous to `ScopedDatatype` for `MPI_Datatype`.
class ScopedOp {
public:
    /// @brief Constructs an empty, non-owning handle (`MPI_OP_NULL`).
    ScopedOp() noexcept : _op(MPI_OP_NULL), _owns(false) {}

    /// @brief Wrap an existing `MPI_Op`.
    /// @param op    The op to wrap.
    /// @param owns  If `true`, `MPI_Op_free` is called on destruction.
    ScopedOp(MPI_Op op, bool owns) noexcept : _op(op), _owns(owns) {}

    ScopedOp(ScopedOp const&)            = delete;
    ScopedOp& operator=(ScopedOp const&) = delete;

    /// @brief Move constructor. Transfers ownership; the moved-from handle no longer frees the op.
    ScopedOp(ScopedOp&& other) noexcept : _op(other._op), _owns(other._owns) {
        other._owns = false;
    }
    /// @brief Move assignment. Frees any currently owned op, then transfers ownership.
    ScopedOp& operator=(ScopedOp&& other) noexcept {
        if (this != &other) {
            _free();
            _op         = other._op;
            _owns       = other._owns;
            other._owns = false;
        }
        return *this;
    }

    ~ScopedOp() {
        _free();
    }

    /// @returns The underlying `MPI_Op`.
    MPI_Op get() const noexcept {
        return _op;
    }

private:
    void _free() noexcept {
        if (_owns) {
            int const err = MPI_Op_free(&_op);
            KAMPING_ASSERT(err == MPI_SUCCESS, "MPI_Op_free failed");
            _owns = false;
        }
    }

    MPI_Op _op;
    bool   _owns;
};

// ---------------------------------------------------------------------------
// ScopedFunctorOp — MPI_Op_create from a default-constructible C++ functor
// ---------------------------------------------------------------------------

/// @brief RAII handle that creates an `MPI_Op` from a default-constructible C++ functor.
///
/// Calls `MPI_Op_create` on construction and `MPI_Op_free` on destruction.
/// The functor is invoked via `MPI_Op_create`'s callback and must be default-constructible
/// (i.e. stateless or state carried via static variables). For capturing lambdas use `ScopedCallbackOp`.
///
/// @tparam is_commutative Whether the operation is commutative.
/// @tparam T              Element type the functor operates on.
/// @tparam Op             Functor type. Must be default-constructible and callable as `T(T const&, T const&)`.
template <bool is_commutative, typename T, typename Op>
class ScopedFunctorOp {
    static_assert(
        std::is_default_constructible_v<Op>,
        "ScopedFunctorOp requires a default-constructible functor. Use ScopedCallbackOp for lambdas."
    );
    static_assert(std::is_invocable_r_v<T, Op, T const&, T const&>, "Op must be callable as T(T const&, T const&).");

public:
    /// @brief Creates an `MPI_Op` for the given functor.
    ScopedFunctorOp(Op op) : _functor(std::move(op)), _op(_make_scoped_op()) {}

    ScopedFunctorOp(ScopedFunctorOp const&)            = delete;
    ScopedFunctorOp& operator=(ScopedFunctorOp const&) = delete;
    ScopedFunctorOp(ScopedFunctorOp&&)                 = delete;
    ScopedFunctorOp& operator=(ScopedFunctorOp&&)      = delete;

    /// @returns The underlying `MPI_Op`. Do not free manually — the destructor does it.
    MPI_Op get() const noexcept {
        return _op.get();
    }

    /// @brief Applies the functor to two values.
    T operator()(T const& lhs, T const& rhs) const {
        return _functor(lhs, rhs);
    }

private:
    /// @brief MPI callback: applies a default-constructed `Op` element-wise.
    static void _execute(void* invec, void* inoutvec, int* len, MPI_Datatype* /*datatype*/) {
        T* in    = static_cast<T*>(invec);
        T* inout = static_cast<T*>(inoutvec);
        std::transform(in, in + *len, inout, inout, Op{});
    }

    static ScopedOp _make_scoped_op() {
        MPI_Op raw;
        MPI_Op_create(_execute, static_cast<int>(is_commutative), &raw);
        return ScopedOp{raw, true};
    }

    Op       _functor;
    ScopedOp _op;
};

// ---------------------------------------------------------------------------
// ScopedCallbackOp — MPI_Op_create from a raw MPI callback function pointer
// ---------------------------------------------------------------------------

/// @brief RAII handle that creates an `MPI_Op` from a raw MPI callback function pointer.
///
/// Calls `MPI_Op_create` on construction and `MPI_Op_free` on destruction.
/// A default-constructed `ScopedCallbackOp` is empty (`MPI_OP_NULL`, non-owning).
/// Supports move construction and assignment; the moved-from handle becomes empty.
///
/// Typically used for lambdas with captures, where the lambda is stored separately and a
/// raw function pointer (via a static trampoline) is passed to `MPI_Op_create`.
///
/// @tparam is_commutative Whether the operation is commutative.
template <bool is_commutative>
class ScopedCallbackOp {
public:
    /// @brief The MPI callback signature expected by `MPI_Op_create`.
    using callback_type = void (*)(void*, void*, int*, MPI_Datatype*);

    /// @brief Constructs an empty, non-owning handle (`MPI_OP_NULL`).
    ScopedCallbackOp() noexcept = default;

    /// @brief Creates an `MPI_Op` for the given callback.
    /// @param ptr Non-null MPI callback function pointer.
    explicit ScopedCallbackOp(callback_type ptr) : _op(_make_scoped_op(ptr)) {
        KAMPING_ASSERT(ptr != nullptr);
    }

    ScopedCallbackOp(ScopedCallbackOp const&)            = delete;
    ScopedCallbackOp& operator=(ScopedCallbackOp const&) = delete;

    /// @brief Move constructor. The moved-from handle becomes empty.
    ScopedCallbackOp(ScopedCallbackOp&&) noexcept = default;
    /// @brief Move assignment. Frees any currently owned op, then takes ownership.
    ScopedCallbackOp& operator=(ScopedCallbackOp&&) noexcept = default;

    /// @returns The underlying `MPI_Op` (`MPI_OP_NULL` if default-constructed). Do not free manually.
    MPI_Op get() const noexcept {
        return _op.get();
    }

private:
    static ScopedOp _make_scoped_op(callback_type ptr) {
        MPI_Op raw;
        MPI_Op_create(ptr, static_cast<int>(is_commutative), &raw);
        return ScopedOp{raw, true};
    }

    ScopedOp _op; // default-constructed: MPI_OP_NULL, non-owning
};

// ---------------------------------------------------------------------------
// with_operation_functor — runtime MPI_Op → functor dispatch
// ---------------------------------------------------------------------------

/// @brief Calls `func` with the functor object corresponding to the given builtin `MPI_Op`.
///
/// For unknown ops, calls `func(kamping::ops::null<>{})`. Useful for implementing
/// `MPI_Reduce_local`-style helpers that need a C++ callable for a runtime `MPI_Op`.
///
/// @tparam Functor  Callable accepting any `kamping::ops::*` functor type.
template <typename Functor>
auto with_operation_functor(MPI_Op op, Functor&& func) {
    if (op == MPI_MAX)
        return func(ops::max<>{});
    else if (op == MPI_MIN)
        return func(ops::min<>{});
    else if (op == MPI_SUM)
        return func(ops::plus<>{});
    else if (op == MPI_PROD)
        return func(ops::multiplies<>{});
    else if (op == MPI_LAND)
        return func(ops::logical_and<>{});
    else if (op == MPI_LOR)
        return func(ops::logical_or<>{});
    else if (op == MPI_LXOR)
        return func(ops::logical_xor<>{});
    else if (op == MPI_BAND)
        return func(ops::bit_and<>{});
    else if (op == MPI_BOR)
        return func(ops::bit_or<>{});
    else if (op == MPI_BXOR)
        return func(ops::bit_xor<>{});
    else
        return func(ops::null<>{});
}

} // namespace kamping::types
