// This file is part of KaMPIng.
//
// Copyright 2021-2025 The KaMPIng Authors
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
/// @brief Type traits and dispatcher for mapping C++ types to MPI datatypes.

#pragma once
#include <type_traits>

#include "kamping/types/builtin_types.hpp"
#include "kamping/types/detail/contiguous_type_fwd.hpp"
#include "kamping/types/detail/type_helpers.hpp"

namespace kamping {

/// @addtogroup kamping_types
/// @{

namespace types {
/// @brief Maps a C++ type \p T to a type trait for constructing an MPI_Datatype.
///
/// | C++ type | Result |
/// |----------|--------|
/// | MPI builtin (`int`, `double`, …, `kabool`) | `builtin_type<T>` |
/// | Enum | dispatches to `type_dispatcher<underlying_type>()` |
/// | `T[N]`, `std::array<T, N>` | `contiguous_type<T, N>` |
/// | Everything else | `internal::no_matching_type` |
///
/// Specialize \ref kamping::mpi_type_traits to handle additional types.
///
/// @returns The corresponding type trait for the type \p T.
template <typename T>
auto type_dispatcher() {
    using T_no_const = std::remove_const_t<T>;

    static_assert(
        !std::is_pointer_v<T_no_const>,
        "MPI does not support pointer types. Why do you want to transfer a pointer over MPI?"
    );
    static_assert(!std::is_function_v<T_no_const>, "MPI does not support function types.");
    static_assert(!std::is_union_v<T_no_const>, "MPI does not support union types.");
    static_assert(!std::is_void_v<T_no_const>, "There is no MPI datatype corresponding to void.");

    if constexpr (is_builtin_type_v<T_no_const>) {
        return builtin_type<T_no_const>{};
    } else if constexpr (std::is_enum_v<T_no_const>) {
        return type_dispatcher<std::underlying_type_t<T_no_const>>();
    } else if constexpr (std::is_array_v<T_no_const>) {
        return contiguous_type<std::remove_extent_t<T_no_const>, std::extent_v<T_no_const>>{};
    } else if constexpr (internal::is_std_array<T_no_const>::value) {
        return contiguous_type<
            typename internal::is_std_array<T_no_const>::value_type,
            internal::is_std_array<T_no_const>::size>{};
    } else {
        return internal::no_matching_type{};
    }
}

/// @brief Whether the type is handled by the auto-dispatcher \ref kamping::types::type_dispatcher().
template <typename T>
static constexpr bool has_auto_dispatched_type_v =
    !std::is_same_v<decltype(type_dispatcher<T>()), internal::no_matching_type>;
} // namespace types

/// @brief The type trait that maps a C++ type \p T to a type trait for constructing an MPI_Datatype.
///
/// The default behavior is controlled by \ref type_dispatcher. Specialize this trait to support
/// additional types.
template <typename T, typename Enable = void>
struct mpi_type_traits {};

/// @brief Partial specialization of \ref mpi_type_traits for types matched by \ref type_dispatcher.
template <typename T>
struct mpi_type_traits<T, std::enable_if_t<types::has_auto_dispatched_type_v<T>>> {
    /// @brief The base type of this trait obtained via \ref type_dispatcher.
    using base = decltype(types::type_dispatcher<T>());
    /// @brief The category of the type.
    static constexpr types::TypeCategory category = base::category;
    /// @brief Whether the type has to be committed before it can be used in MPI calls.
    static constexpr bool has_to_be_committed = types::category_has_to_be_committed(category);
    /// @brief The MPI_Datatype corresponding to the type T.
    static MPI_Datatype data_type() {
        return decltype(types::type_dispatcher<T>())::data_type();
    }
};

/// @brief Check if the type has a static type definition, i.e. \ref mpi_type_traits is defined.
template <typename, typename Enable = void>
struct has_static_type : std::false_type {};

/// @brief Check if the type has a static type definition, i.e. \ref mpi_type_traits is defined.
template <typename T>
struct has_static_type<T, std::void_t<decltype(mpi_type_traits<T>::data_type())>> : std::true_type {};

/// @brief `true` if \ref mpi_type_traits provides a `data_type()` function.
template <typename T>
static constexpr bool has_static_type_v = has_static_type<T>::value;

/// @}

} // namespace kamping
