#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include "kamping/v2/ranges/view_interface.hpp"

namespace kamping::ranges {

/// Non-owning view over an lvalue. T is unconstrained — supports both ranges and
/// non-range buffer types (e.g. a struct with mpi_data()/mpi_size() but no begin()/end()).
/// All MPI protocol methods (mpi_data, mpi_size, mpi_type, and future mpi_resize_for_receive)
/// are forwarded automatically via view_interface.
template <typename T>
class ref_view : public view_interface<ref_view<T>> {
    T* r_;

public:
    constexpr explicit ref_view(T& r) noexcept : r_(std::addressof(r)) {}

    constexpr T& base() const noexcept {
        return *r_;
    }
};

/// Owning view over an rvalue. Moves the object in and owns it.
/// Same protocol forwarding as ref_view via view_interface.
template <std::movable T>
class owning_view : public view_interface<owning_view<T>> {
    T r_;

public:
    constexpr explicit owning_view(T&& r) noexcept(std::is_nothrow_move_constructible_v<T>)
        : r_(std::move(r)) {}

    constexpr T&       base() &      noexcept { return r_; }
    constexpr T const& base() const& noexcept { return r_; }
    constexpr T&&      base() &&     noexcept { return std::move(r_); }
};

/// Wraps a value in the appropriate view:
///   - already a kamping view (inherits view_interface_base) → pass through unchanged
///   - lvalue reference                                       → ref_view
///   - rvalue                                                 → owning_view
template <typename R>
constexpr auto all(R&& r) {
    if constexpr (std::derived_from<std::remove_cvref_t<R>, view_interface_base>)
        return std::forward<R>(r);
    else if constexpr (std::is_lvalue_reference_v<R>)
        return ref_view<std::remove_reference_t<R>>{r};
    else
        return owning_view<std::remove_cvref_t<R>>{std::forward<R>(r)};
}

template <typename R>
using all_t = decltype(all(std::declval<R>()));

} // namespace kamping::ranges
