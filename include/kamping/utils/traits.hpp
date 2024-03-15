// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once
#include <array> // defines std::tuple_size for std::array
#include <tuple> // defines std::tuple_size for tuples
#include <type_traits>
#include <utility> // defines std::tuple_size for std::pair

namespace kamping {

/// @brief A type trait that checks if a type \p T is a range, i.e., it has `std::begin` and `std::end` defined.
template <typename T>
struct is_range {
    template <typename S>
    /// @brief Only enable this overload if `std::begin` and `std::end` are defined for \p S.
    static auto test(int) -> decltype(std::begin(std::declval<S>()), std::end(std::declval<S>()), std::true_type{});
    template <typename>
    /// @brief Fallback overload.
    static auto           test(...) -> std::false_type;
    static constexpr bool value = decltype(test<T>(0))::value; ///< The value of the trait.
};

/// @brief A type trait that checks if a type \p T is a range, i.e., it has `std::begin` and `std::end` defined.
template <typename T>
constexpr bool is_range_v = is_range<T>::value;

/// @brief A type trait that checks if a type \p T is a contiguous and sized range, i.e., it is a range and has
/// `std::size` and `std::data` defined.
template <typename T>
struct is_contiguous_sized_range {
    /// @brief Only enable this overload if `std::size` and `std::data` are defined for \p S.
    template <typename S>
    static auto test(int) -> decltype(std::size(std::declval<S>()), std::data(std::declval<S>()), std::true_type{});
    /// @brief Fallback overload.
    template <typename>
    static auto           test(...) -> std::false_type;
    static constexpr bool value = decltype(test<T>(0))::value; ///< The value of the trait.
};

/// @brief A type trait that checks if a type \p T is a contiguous and sized range, i.e., it is a range and has
/// `std::size` and `std::data` defined.
template <typename T>
constexpr bool is_contiguous_sized_range_v = is_contiguous_sized_range<T>::value;

/// @brief A type trait that checks if a type \p T is a pair-like type, i.e., it may be destructured using `std::get<0>`
/// and `std::get<1>` and has a size of 2.
template <typename T>
struct is_pair_like {
    /// @brief Only enable this overload if `std::tuple_size` is defined for \p S.
    template <typename S>
    static auto test(int) -> decltype(std::integral_constant<size_t, std::tuple_size<S>::value>{});
    template <typename>
    /// @brief Fallback overload, returns size 0.
    static auto           test(...) -> std::integral_constant<size_t, 0>;
    static constexpr bool value = decltype(test<T>(0))::value == 2; ///< The value of the trait.
};

/// @brief A type trait that checks if a type \p T is a pair-like type, i.e., it may be destructured using `std::get<0>`
/// and `std::get<1>` and has a size of 2.
template <typename T>
constexpr bool is_pair_like_v = is_pair_like<T>::value;

/// @brief A type trait that checks if a type T a pair-like type using \c is_pair_like, the first element is
/// convertible to int, and the second element satisfies \c is_contiguous_sized_range_v.
template <typename T>
constexpr bool is_destination_buffer_pair_v = [] {
    if constexpr (is_pair_like_v<T>) {
        return is_contiguous_sized_range_v<std::remove_const_t<std::tuple_element_t<
                   1,
                   T>>> && std::is_convertible_v<std::remove_const_t<std::tuple_element_t<0, T>>, int>;
    } else {
        return false;
    }
}();

/// @brief A type trait that checks if a type \p T is a sparse send buffer, i.e., it is a range of pair-like which are
/// (dst, message) pairs. (see \c is_destination_buffer_pair_v)
template <typename T>
constexpr bool is_sparse_send_buffer_v = [] {
    if constexpr (is_range_v<T>) {
        return is_destination_buffer_pair_v<std::remove_const_t<typename T::value_type>>;
    } else {
        return false;
    }
}();

/// @brief A type traits that checks if a type is a nested send buffer, i.e., it is a range of contiguous ranges (see
/// \c is_contiguous_sized_range_v).
template <typename T>
constexpr bool is_nested_send_buffer_v = [] {
    if constexpr (is_range_v<T>) {
        return is_contiguous_sized_range_v<std::remove_const_t<typename T::value_type>>;
    } else {
        return false;
    }
}();
} // namespace kamping
