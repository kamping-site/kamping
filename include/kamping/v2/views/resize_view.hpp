#pragma once

#include <cstddef>

#include "kamping/v2/ranges/adaptor_closure.hpp"
#include "kamping/v2/ranges/all.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/ranges/view_interface.hpp"

namespace kamping::ranges {

/// Wraps a resizable container and defers the actual resize until mpi_data() is first accessed.
/// set_recv_count(n) is called by the collective after inferring n via infer(). mpi_size()
/// returns the stored count immediately; mpi_data() triggers the resize on first call.
/// mpi_type() is forwarded from the base via view_interface.
template <typename Base>
class resize_buf_view : public view_interface<resize_buf_view<Base>> {
    Base           base_;
    std::ptrdiff_t recv_count_ = 0;
    bool           needs_resize_ = false;

public:
    explicit resize_buf_view(Base base) : base_(std::move(base)) {}

    constexpr Base&       base() &      noexcept { return base_; }
    constexpr Base const& base() const& noexcept { return base_; }

    /// Called by the collective with the inferred recv count. Does not resize yet.
    void set_recv_count(std::ptrdiff_t n) {
        recv_count_  = n;
        needs_resize_ = true;
    }

    /// Returns the recv count set by set_recv_count(). Overrides view_interface::mpi_size().
    std::ptrdiff_t mpi_size() const {
        return recv_count_;
    }

    /// Triggers the lazy resize on first access, then returns the data pointer.
    /// Overrides view_interface::mpi_data().
    auto mpi_data() {
        if (needs_resize_) {
            kamping::ranges::resize_for_receive(base_, recv_count_);
            needs_resize_ = false;
        }
        return kamping::ranges::data(base_);
    }
};

template <typename R>
resize_buf_view(R&&) -> resize_buf_view<kamping::ranges::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<resize_buf_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::ranges

namespace kamping::views {

/// Wraps a resizable buffer so the collective can call set_recv_count(n) to infer its size.
/// Always outermost in a pipe chain: vec | with_type(t) | resize
/// Does not take extra arguments — use as: val | views::resize  or  views::resize(val)
inline constexpr struct resize_fn : kamping::ranges::adaptor_closure<resize_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::ranges::resize_buf_view(
            kamping::ranges::all(std::forward<R>(r))
        );
    }
} resize{};

} // namespace kamping::views
