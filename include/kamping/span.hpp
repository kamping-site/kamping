// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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

#include <type_traits>

namespace kamping::internal {
///
/// @brief Obtain the address represented by \c p. Modelled after C++20's \c std::to_address.
/// See https://en.cppreference.com/w/cpp/memory/to_address for details.
/// @param p a raw pointer
/// @tparam the underlying type
template <typename T>
constexpr T* to_address(T* p) noexcept {
    static_assert(!std::is_function_v<T>);
    return p;
}

/// @brief Obtain the address represented by \c p. Modelled after C++20's \c std::to_address.
/// See https://en.cppreference.com/w/cpp/memory/to_address for details.
/// @param p a smart pointer
/// @tparam the pointer type
template <typename T>
constexpr auto to_address(T const& p) noexcept {
    // specialization to make this work with smart pointers
    // the standard mandates the pointer to be obtained in this way
    return to_address(p.operator->());
}
} // namespace kamping::internal

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) || __cplusplus >= 202002L) // C++ 20

    #include <span>

namespace kamping {

// std::span is only available in C++ 20 and upwards.
template <typename T>
using Span = std::span<T>;

} // namespace kamping

#else // C++ 17

    #include <cstddef>
    #include <tuple>

namespace kamping {

/// @brief A span modeled after C++20's \c std::span.
///
/// Since KaMPIng needs to be C++17 compatible and \c std::span is part of C++20, we need our own implementation of
/// the above-described functionality.
/// @tparam T type for which the span is defined.
template <typename T>
class Span {
public:
    using element_type    = T;                   ///< Element type; i.e. \c T.
    using value_type      = std::remove_cv_t<T>; ///< Value type; i.e. \c T with volatile and const qualifiers removed.
    using size_type       = size_t;              ///< The type used for the size of the span.
    using difference_type = std::ptrdiff_t;      ///< The type used for the difference between two elements in the span.
    using pointer         = T*;                  ///< The type of a pointer to a single elements in the span.
    using const_pointer   = T const*;            ///< The type of a const pointer to a single elements in the span.
    using reference       = T&;                  ///< The type of a reference to a single elements in the span.
    using const_reference = T const&;            ///< The type of a const reference to a single elements in the span.

    /// @brief Constructor for a span from a pointer and a size.
    ///
    /// @param ptr Pointer to the first element in the span.
    /// @param size The number of elements in the span.
    constexpr Span(pointer ptr, size_type size) : _ptr(ptr), _size(size) {}

    /// @brief Constructs a span that is a view over the range <code>[first, last)</code>; the resulting span has
    /// <code>data() == kamping::internal::to_address(first)</code> and <code>size() == last-first</code>.
    ///
    /// If <code>[first, last)</code> is not a valid range, or if \c It does not model a C++20 contiguous iterator, the
    /// behavior is undefined. This is analagous to the behavior of \c std::span.
    /// @param first begin iterator of the range
    /// @param last end iterator of the range
    /// @tparam It the iterator type.
    template <typename It>
    constexpr Span(It first, It last)
        : _ptr(internal::to_address(first)),
          _size(static_cast<size_type>(last - first)) {}

    /// @brief Get access to the underlying memory.
    ///
    /// @return Pointer to the underlying memory.
    constexpr pointer data() const noexcept {
        return _ptr;
    }

    /// @brief Returns the number of elements in the Span.
    ///
    /// @return Number of elements in the span.
    constexpr size_type size() const noexcept {
        return _size;
    }

    /// @brief Return the number of bytes occupied by the elements in the Span.
    ///
    /// @return The number of elements in the span times the number of bytes per element.
    constexpr size_type size_bytes() const noexcept {
        return _size * sizeof(value_type);
    }

    /// @brief Check if the Span is empty.
    ///
    /// @return \c true if the Span is empty, \c false otherwise.
    [[nodiscard]] constexpr bool empty() const noexcept {
        return _size == 0;
    }

protected:
    pointer   _ptr;  ///< Pointer to the data referred to by Span.
    size_type _size; ///< Number of elements of type T referred to by Span.
};

} // namespace kamping

#endif // C++ 17
