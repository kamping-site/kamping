#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include <mpi.h>

#include "kamping/builtin_types.hpp"

namespace kamping::ranges {

// Tag base for all kamping views. Used to guard the std::ranges::size fallback
// against ADL circularity (std::ranges::size → ADL kamping::ranges::size → std::ranges::sized_range).
struct view_interface_base {};

template <class T>
concept integer_like = std::integral<T> && !std::same_as<T, bool>;

template <class T>
concept ptr_to_object = std::is_pointer_v<T>
                        && (std::is_object_v<std::remove_pointer_t<T>> || std::is_void_v<std::remove_pointer_t<T>>);

template <typename T>
concept has_mpi_compatible_size_member = requires(std::remove_reference_t<T> const& t) {
    { t.mpi_size() } -> integer_like;
};

template <typename T>
concept has_mpi_compatible_data_member = requires(T&& t) {
    { t.mpi_data() } -> ptr_to_object;
};

template <typename T>
concept has_mpi_compatible_type_member = requires(std::remove_reference_t<T> const& t) {
    { t.mpi_type() } -> std::convertible_to<MPI_Datatype>;
};

template <typename T>
concept range_of_builtin_mpi_type =
    std::ranges::range<T> && kamping::is_builtin_type_v<std::remove_cvref_t<std::ranges::range_value_t<T>>>;

/// Type implements the custom MPI resize protocol (preferred over plain resize()).
template <typename T>
concept has_mpi_resize_for_receive = requires(T& t, std::ptrdiff_t n) {
    t.mpi_resize_for_receive(n);
};

/// Type is a standard resizable container (e.g. std::vector).
template <typename T>
concept has_resize = requires(T& t, std::size_t n) { t.resize(n); };

// ──────────────────────────────────────────────────────────────────────────────
// buffer_traits — trait class for non-intrusive MPI buffer customization.
//
// Specialize buffer_traits<T> for types you don't own (or want to selectively
// override). Only implement the members you need; unimplemented members fall
// back to the corresponding mpi_size() / mpi_data() / mpi_type() member
// functions.
//
//   template <>
//   struct kamping::ranges::buffer_traits<my_struct> {
//       static std::ptrdiff_t size(my_struct const& t) { return 1; }
//       static int const*     data(my_struct const& t) { return &t.val; } // send
//       static int*           data(my_struct& t)       { return &t.val; } // recv
//       static MPI_Datatype   type(my_struct const& t) { return MPI_INT; }
//   };
// ──────────────────────────────────────────────────────────────────────────────
template <typename T>
struct buffer_traits {};

/// buffer_traits<T> provides a size(T const&) returning an integer-like type.
template <typename T>
concept buffer_traits_has_size = requires(T const& t) {
    { buffer_traits<T>::size(t) } -> integer_like;
};

/// buffer_traits<T> provides a data(T const&) returning a pointer (for send).
template <typename T>
concept buffer_traits_has_const_data = requires(T const& t) {
    { buffer_traits<T>::data(t) } -> ptr_to_object;
};

/// buffer_traits<T> provides a data(T&) returning a pointer (for recv).
template <typename T>
concept buffer_traits_has_data = requires(T& t) {
    { buffer_traits<T>::data(t) } -> ptr_to_object;
};

/// buffer_traits<T> provides a type(T const&) returning an MPI_Datatype.
template <typename T>
concept buffer_traits_has_type = requires(T const& t) {
    { buffer_traits<T>::type(t) } -> std::convertible_to<MPI_Datatype>;
};

// ──────────────────────────────────────────────────────────────────────────────
// size() dispatch — priority: buffer_traits > mpi_size() member > std::ranges::size
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
constexpr auto size(T&&) = delete;

template <typename T>
    requires buffer_traits_has_size<std::remove_cvref_t<T>>
constexpr auto size(T&& t) {
    return buffer_traits<std::remove_cvref_t<T>>::size(t);
}

template <typename T>
    requires (!buffer_traits_has_size<std::remove_cvref_t<T>>)
          && has_mpi_compatible_size_member<T>
constexpr auto size(T&& t) {
    return t.mpi_size();
}

// The !derived_from guard prevents evaluating std::ranges::sized_range for kamping view
// types, breaking the ADL cycle: std::ranges::size → ADL kamping::ranges::size
// → std::ranges::sized_range → std::ranges::size → … (circular hard error).
template <typename T>
    requires (!buffer_traits_has_size<std::remove_cvref_t<T>>)
          && (!has_mpi_compatible_size_member<T>)
          && (!std::derived_from<std::remove_cvref_t<T>, view_interface_base>)
          && std::ranges::sized_range<T>
constexpr auto size(T&& t) {
    return std::ranges::size(std::forward<T>(t));
}

// ──────────────────────────────────────────────────────────────────────────────
// data() dispatch — priority: buffer_traits > mpi_data() member > std::ranges::data
//
// buffer_traits dispatch uses std::forward to preserve const-qualifiers, so
// buffer_traits<T> can provide both a const overload (send) and a non-const
// overload (recv) and the right one is selected at the call site.
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires (buffer_traits_has_const_data<std::remove_cvref_t<T>> || buffer_traits_has_data<std::remove_cvref_t<T>>)
constexpr auto data(T&& t) {
    return buffer_traits<std::remove_cvref_t<T>>::data(std::forward<T>(t));
}

template <typename T>
    requires (!buffer_traits_has_const_data<std::remove_cvref_t<T>>)
          && (!buffer_traits_has_data<std::remove_cvref_t<T>>)
          && has_mpi_compatible_data_member<T>
constexpr auto data(T&& t) {
    return t.mpi_data();
}

// std::ranges::data has no ADL step so no circular dependency risk, but the
// !derived_from guard is added for consistency with the size() overload.
template <typename T>
    requires (!buffer_traits_has_const_data<std::remove_cvref_t<T>>)
          && (!buffer_traits_has_data<std::remove_cvref_t<T>>)
          && (!has_mpi_compatible_data_member<T>)
          && (!std::derived_from<std::remove_cvref_t<T>, view_interface_base>)
          && std::ranges::contiguous_range<T>
constexpr auto data(T&& t) {
    return std::ranges::data(std::forward<T>(t));
}

// ──────────────────────────────────────────────────────────────────────────────
// type() dispatch — priority: buffer_traits > mpi_type() member > builtin deduction
// ──────────────────────────────────────────────────────────────────────────────

template <typename T>
    requires buffer_traits_has_type<std::remove_cvref_t<T>>
constexpr auto type(T&& t) {
    return buffer_traits<std::remove_cvref_t<T>>::type(t);
}

template <typename T>
    requires (!buffer_traits_has_type<std::remove_cvref_t<T>>)
          && has_mpi_compatible_type_member<T>
constexpr auto type(T&& t) {
    return t.mpi_type();
}

template <typename T>
    requires (!buffer_traits_has_type<std::remove_cvref_t<T>>)
          && (!has_mpi_compatible_type_member<T>)
          && range_of_builtin_mpi_type<T>
constexpr auto type(T&& /* t */) {
    return builtin_type<std::remove_cvref_t<std::ranges::range_value_t<T>>>::data_type();
}

/// Resize t to hold n MPI elements before a receive. Dispatches to:
///   1. t.mpi_resize_for_receive(n) — custom protocol (e.g. resize_and_overwrite, NUMA alloc)
///   2. t.resize(n)                 — standard containers
template <has_mpi_resize_for_receive T>
void resize_for_receive(T& t, std::ptrdiff_t n) {
    t.mpi_resize_for_receive(n);
}

template <typename T>
    requires (!has_mpi_resize_for_receive<T>) && has_resize<T>
void resize_for_receive(T& t, std::ptrdiff_t n) {
    t.resize(static_cast<std::size_t>(n));
}

} // namespace kamping::ranges
