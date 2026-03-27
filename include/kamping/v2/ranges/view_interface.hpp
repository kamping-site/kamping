#pragma once

#include <ranges>

#include "concepts.hpp"
#include "ranges.hpp"

namespace kamping::ranges {
template <typename Derived>
struct view_interface {
    constexpr Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    constexpr Derived const& derived() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    constexpr auto begin() const{
      return std::ranges::begin(derived().base());
    }

    constexpr auto end() const {
      return std::ranges::end(derived().base());
    }

    template <typename _Derived = Derived>
    auto mpi_type() const
        requires kamping::ranges::has_mpi_type<decltype(derived().base())>
    {
        return kamping::ranges::type(derived().base());
    }
    // constexpr auto& base() & noexcept {
    //   return static_cast<Derived&>(*this).base_;
    //   }
};
} // namespace kamping::ranges
