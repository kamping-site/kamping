#pragma once

#include <ranges>

#include "concepts.hpp"
#include "ranges.hpp"

namespace kamping::ranges {
namespace detail {
template <typename D>
concept has_base_range = requires(D& d) {
    { d.base() } -> std::ranges::range;
};

template <typename D>
concept has_const_base_range = requires(D const& d) {
    { d.base() } -> std::ranges::range;
};
} // namespace detail

struct view_interface_base {};

template <typename Derived>
struct view_interface : public view_interface_base, public std::ranges::view_interface<Derived> {
    constexpr Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    constexpr Derived const& derived() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    constexpr auto begin()
        requires detail::has_base_range<Derived>
    {
        return std::ranges::begin(derived().base());
    }

    constexpr auto end()
        requires detail::has_base_range<Derived>
    {
        return std::ranges::end(derived().base());
    }

    constexpr auto begin() const
        requires detail::has_const_base_range<Derived>
    {
        return std::ranges::begin(derived().base());
    }

    constexpr auto end() const
        requires detail::has_const_base_range<Derived>
    {
        return std::ranges::end(derived().base());
    }

    template <typename _Derived = Derived>
    auto mpi_type() const
        requires kamping::ranges::has_mpi_type<decltype(derived().base())>
    {
        return kamping::ranges::type(derived().base());
    }

    constexpr auto mpi_size() const
        requires kamping::ranges::has_mpi_size<decltype(derived().base())>
    {
        return kamping::ranges::size(derived().base());
    }

    constexpr auto mpi_data()
        requires kamping::ranges::has_mpi_data<decltype(derived().base())>
    {
        return kamping::ranges::data(derived().base());
    }

    constexpr auto mpi_data() const
        requires kamping::ranges::has_mpi_data<decltype(derived().base())>
    {
        return kamping::ranges::data(derived().base());
    }

    void mpi_resize_for_receive(std::ptrdiff_t n)
        requires(kamping::ranges::has_mpi_resize_for_receive<decltype(derived().base())>
                 || kamping::ranges::has_resize<decltype(derived().base())>)
    {
        kamping::ranges::resize_for_receive(derived().base(), n);
    }
};
} // namespace kamping::ranges

/* template <typename Derived> */
/* inline constexpr bool std::ranges::enable_borrowed_range<kamping::ranges::view_interface<Derived>> = */
/*     std::ranges::borrowed_range<decltype(std::declval<Derived&>().base())>; */
