// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
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

#include <cstddef>
#include <iterator>
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
    using iterator        = pointer;             ///< The type of an iterator to a single elements in the span.
    using reverse_iterator =
        std::reverse_iterator<iterator>; ///< The type of a reverse iterator to a single elements in the span.

    /// @brief Default constructor for an empty span. The pointer is set to \c nullptr and the size to 0.
    constexpr Span() noexcept : _ptr(nullptr), _size(0) {}

    /// @brief Constructor for a span from an iterator of type \p It and a \p size.
    ///
    /// @param first Iterator pointing to the first element of the span.
    /// @param size The number of elements in the span.
    /// @tparam It The iterator type.
    template <typename It>
    constexpr Span(It first, size_type size) : _ptr(internal::to_address(first)),
                                               _size(size) {}

    /// @brief Constructs a span that is a view over the range <code>[first, last)</code>; the resulting span has
    /// <code>data() == kamping::internal::to_address(first)</code> and <code>size() == last-first</code>.
    ///
    /// If <code>[first, last)</code> is not a valid range, or if \c It does not model a C++20 contiguous iterator, the
    /// behavior is undefined. This is analogous to the behavior of \c std::span.
    /// @param first begin iterator of the range
    /// @param last end iterator of the range
    /// @tparam It the iterator type.
    template <typename It>
    constexpr Span(It first, It last)
        : _ptr(internal::to_address(first)),
          _size(static_cast<size_type>(last - first)) {}

    /// @brief Constructs a span that is a view over the range \c range. The resulting span has
    /// <code>data() == std::data(range)</code> and <code>size() == std::size()</code>.
    ///
    /// If <code>range</code> does not model a C++20 contiguous range, the
    /// behavior is undefined. This is analogous to the behavior of \c std::span.
    /// @param range The range.
    /// @tparam Range The range type.
    template <typename Range>
    constexpr Span(Range&& range) : _ptr(std::data(range)),
                                    _size(std::size(range)) {}

    /// @brief Get access to the underlying memory.
    ///
    /// @return Pointer to the underlying memory.
    constexpr pointer data() const noexcept {
        return _ptr;
    }

    /// @brief Get iterator pointing to the first element of the span.
    ///
    /// @return Iterator pointing to the first element of the span.
    constexpr iterator begin() const noexcept {
        return _ptr;
    }

    /// @brief Get iterator pointing past the last element of the span.
    ///
    /// @return Iterator pointing past the last element of the span.
    constexpr iterator end() const noexcept {
        return _ptr + size();
    }

    /// @brief Get a reverse iterator pointing to the first element of the reversed span.
    constexpr reverse_iterator rbegin() const noexcept {
        return std::reverse_iterator{_ptr + _size};
    }

    /// @brief Get a reverse iterator pointing to the last element of the reversed span.
    constexpr reverse_iterator rend() const noexcept {
        return std::reverse_iterator{_ptr};
    }

    /// @brief Access the first element of the span.
    constexpr reference front() const noexcept {
        return *_ptr;
    }

    /// @brief Access the last element of the span.
    constexpr reference back() const noexcept {
        return *(_ptr + _size - 1);
    }

    /// @brief Access the element at index \p idx.
    constexpr reference operator[](size_type idx) const noexcept {
        return _ptr[idx];
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

    /// @brief Obtain a span that is a view over the first \p count elements of the span.
    constexpr Span first(size_type count) const {
        return Span{_ptr, count};
    }

    /// @brief Obtain a span that is a view over the last \p count elements of the span.
    constexpr Span last(size_type count) const {
        return Span{_ptr + _size - count, count};
    }

    /// @brief Obtain a span that is a view over the span elements in the range <code>[offset, offset + count)</code>.
    /// @param offset The offset of the first element of the span.
    /// @param count The number of elements in the span.
    constexpr Span subspan(size_type offset, size_type count) const {
        return Span{_ptr + offset, count};
    }

protected:
    pointer   _ptr;  ///< Pointer to the data referred to by Span.
    size_type _size; ///< Number of elements of type T referred to by Span.
};

// Deduction guides

template <typename Range>
Span(Range&&) -> Span<typename std::remove_reference_t<Range>::value_type>;

template <typename It>
Span(It, It) -> Span<std::remove_reference_t<typename std::iterator_traits<It>::reference> >;

template <typename It>
Span(It, size_t) -> Span<std::remove_reference_t<typename std::iterator_traits<It>::reference> >;
} // namespace kamping
