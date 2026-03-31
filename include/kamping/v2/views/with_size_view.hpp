#pragma once

#include <cstddef>
#include <ranges>

#include "kamping/v2/ranges/adaptor.hpp"
#include "kamping/v2/ranges/all.hpp"
#include "kamping/v2/ranges/view_interface.hpp"

namespace kamping {
namespace ranges {

template <typename Base>
class with_size_view : public kamping::ranges::view_interface<with_size_view<Base>> {
    Base           base_;
    std::ptrdiff_t size_;

public:
    constexpr Base const& base() const& {
        return base_;
    }
    constexpr Base& base() & {
        return base_;
    }

    with_size_view(Base base, std::ptrdiff_t size) : base_(std::move(base)), size_(size) {}

    std::ptrdiff_t mpi_size() const {
        return size_;
    }
};

template <typename R>
with_size_view(R&&, std::ptrdiff_t) -> with_size_view<kamping::ranges::all_t<R>>;

} // namespace ranges

namespace views {

// Useful for non-range objects that expose mpi_data() but no size — compose with with_type
// to build a complete data_buffer. For limiting the element count of a range, prefer std::views::take.
inline constexpr kamping::ranges::adaptor<1, decltype([](auto&& r, std::ptrdiff_t size) {
    return kamping::ranges::with_size_view(kamping::ranges::all(std::forward<decltype(r)>(r)), size);
})> with_size{};

} // namespace views

} // namespace kamping
