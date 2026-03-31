#pragma once
#include <ranges>

#include <mpi.h>

#include "kamping/v2/ranges/adaptor.hpp"
#include "kamping/v2/ranges/all.hpp"
#include "kamping/v2/ranges/view_interface.hpp"

namespace kamping {
namespace ranges {
template <typename Base>
class with_type_view : public kamping::ranges::view_interface<with_type_view<Base>> {
    Base         base_;
    MPI_Datatype type_;

public:
    constexpr Base const& base() const& {
        return base_;
    }
    constexpr Base& base() & {
        return base_;
    }

    with_type_view(Base base, MPI_Datatype type) : base_(std::move(base)), type_(type) {}

    auto mpi_type() const {
        return type_;
    }
};

template <typename R>
with_type_view(R&&, MPI_Datatype) -> with_type_view<kamping::ranges::all_t<R>>;

} // namespace ranges

namespace views {

inline constexpr kamping::ranges::adaptor<1, decltype([](auto&& r, MPI_Datatype type) {
    return kamping::ranges::with_type_view(kamping::ranges::all(std::forward<decltype(r)>(r)), type);
})> with_type{};

} // namespace views

} // namespace kamping
