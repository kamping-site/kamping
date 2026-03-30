#pragma once

#include <ranges>

#include "concepts.hpp"
#include "ranges.hpp"

namespace kamping::ranges {
struct view_interface_base {};

template <typename Derived>
struct view_interface : public std::ranges::view_interface<Derived> {
    constexpr Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    constexpr Derived const& derived() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    constexpr auto begin() const
        requires std::ranges::range<decltype(derived().base())>
    {
        return std::ranges::begin(derived().base());
    }

    constexpr auto end() const
        requires std::ranges::range<decltype(derived().base())>
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
};
} // namespace kamping::ranges
