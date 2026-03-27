#pragma once
#include <concepts>
#include <ranges>

#include <mpi.h>

#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/ranges/view_interface.hpp"

namespace kamping::ranges {

template <typename Base>
class with_type_view : public kamping::ranges::view_interface<with_type_view<Base>>,
                       public std::ranges::view_interface<with_type_view<Base>> {
    Base         base_;
    MPI_Datatype type_;

public:
    constexpr Base base() const&
        requires std::copy_constructible<Base>
    {
        return base_;
    }
    constexpr Base base() && {
        return std::move(base_);
    }

    with_type_view(Base base, MPI_Datatype type) : base_(std::move(base)), type_(type) {}

    auto mpi_type() const {
        return type_;
    }
};

template <typename T>
static constexpr bool mpi_enabled<with_type_view<T>> = true;

template <typename R>
with_type_view(R&&, MPI_Datatype) -> with_type_view<std::views::all_t<R>>;

} // namespace kamping::ranges
