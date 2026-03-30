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
/// @brief Internal type helpers for the kamping-types module.

#pragma once
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace kamping::internal {

/// @brief Helper to check if a type is a `std::pair`.
template <typename T>
struct is_std_pair : std::false_type {};
/// @brief Helper to check if a type is a `std::pair`.
template <typename T1, typename T2>
struct is_std_pair<std::pair<T1, T2>> : std::true_type {};

/// @brief Helper to check if a type is a `std::tuple`.
template <typename T>
struct is_std_tuple : std::false_type {};
/// @brief Helper to check if a type is a `std::tuple`.
template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

/// @brief Helper to check if a type is a `std::array`.
template <typename A>
struct is_std_array : std::false_type {};

/// @brief Helper to check if a type is a `std::array`.
template <typename T, size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {
    using value_type             = T; ///< The type of the elements in the array.
    static constexpr size_t size = N; ///< The number of elements in the array.
};

/// @brief Type tag for indicating that no static type definition exists for a type.
struct no_matching_type {};

} // namespace kamping::internal
