#include <ranges>
#include <utility>
#include <vector>

#include "kamping/data_buffers/data_buffer_concepts.hpp"

template <std::ranges::contiguous_range R>
struct auto_displs_view : std::ranges::view_interface<auto_displs_view<R>> {
    R base_;

    explicit auto_displs_view(R base
    ) requires kamping::HasDisplacements<R> && kamping::HasSetDisplacements<R> && kamping::HasSizeV<R> : base_(base) {
        auto   counts = base_.size_v();
        size_t ranks  = std::ranges::size(counts);
        // Counts have to be of correct size
        std::vector<int> displacements(ranks);
        std::exclusive_scan(
            counts.begin(),
            counts.begin() + kamping::asserting_cast<int>(ranks),
            displacements.begin(),
            0
        );
        base_.set_displacements(std::move(displacements));
    }

    auto displacements() {
        return base_.displacements();
    }
    auto size_v() {
        return base_.size_v();
    }
    auto begin() noexcept {
        return std::ranges::begin(base_);
    }
    auto end() noexcept {
        return std::ranges::end(base_);
    }
    auto data() {
        return std::ranges::data(base_);
    }

    void set_size(std::size_t n) requires kamping::HasSetSize<R> {
        base_.set_size(n);
    }
};

struct auto_displs : std::ranges::range_adaptor_closure<auto_displs> {
    explicit auto_displs() = default;

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return auto_displs_view(std::forward<R>(r));
    }
};

template <std::ranges::contiguous_range R>
struct resize_ext_view : std::ranges::view_interface<resize_ext_view<R>> {
    R base_;

    explicit resize_ext_view(R base
    ) requires kamping::HasDisplacements<R> && kamping::HasSetSize<R> && kamping::HasSizeV<R> : base_(base) {
        auto displs = base_.displacements();
        auto counts = base_.size_v();

        auto counts_ptr = std::ranges::data(displs);
        auto displs_ptr = std::ranges::data(counts);

        size_t ranks = std::ranges::size(counts);

        int recv_buf_size = 0;
        for (size_t i = 0; i < ranks; ++i) {
            recv_buf_size = std::max(recv_buf_size, *(counts_ptr + i) + *(displs_ptr + i));
        }

        base_.set_size(kamping::asserting_cast<size_t>(recv_buf_size));
    }

    auto displacements() {
        return base_.displacements();
    }
    auto size_v() {
        return base_.size_v();
    }
    auto begin() noexcept {
        return std::ranges::begin(base_);
    }
    auto end() noexcept {
        return std::ranges::end(base_);
    }
    auto data() {
        return std::ranges::data(base_);
    }
};

struct resize_ext : std::ranges::range_adaptor_closure<resize_ext> {
    explicit resize_ext() = default;

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return resize_ext_view(std::forward<R>(r));
    }
};