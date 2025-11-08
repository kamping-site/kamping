#pragma once

#include <mdspan>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"

template <std::ranges::contiguous_range R>
struct auto_displs_view : pipe_view_interface<auto_displs_view<R>, R> {
    R base_;

    explicit auto_displs_view(R base) requires kamping::HasDispls<R> && kamping::HasSetDispls<R> : base_(base) {}

    auto displs() {
        if (!displs_set) {
            auto   counts = base_.size_v();
            size_t ranks  = std::ranges::size(counts);
            // Counts have to be of correct size
            std::vector<int> displs(ranks);
            std::exclusive_scan(
                counts.begin(),
                counts.begin() + kamping::asserting_cast<int>(ranks),
                displs.begin(),
                0
            );
            base_.set_displs(std::move(displs));
            displs_set = true;
        }
        return base_.displs();
    }

    bool displs_set = false;
};

struct auto_displs : std::ranges::range_adaptor_closure<auto_displs> {
    explicit auto_displs() = default;

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return auto_displs_view(std::forward<R>(r));
    }
};

template <std::ranges::contiguous_range R>
struct resize_ext_view : pipe_view_interface<resize_ext_view<R>, R> {
    R base_;

    explicit resize_ext_view(R base) requires kamping::HasDispls<R> && kamping::HasSetSize<R> && kamping::HasSizeV<R>
        : base_(base) {}

    auto data() {
        resize();
        return std::ranges::data(base_);
    }
    auto size() {
        resize();
        return std::ranges::size(base_);
    }

    void resize() {
        if (!resized) {
            auto displs = base_.displs();
            auto counts = base_.size_v();

            auto counts_ptr = std::ranges::data(displs);
            auto displs_ptr = std::ranges::data(counts);

            size_t ranks = std::ranges::size(counts);

            int recv_buf_size = 0;
            for (size_t i = 0; i < ranks; ++i) {
                recv_buf_size = std::max(recv_buf_size, *(counts_ptr + i) + *(displs_ptr + i));
            }

            base_.set_size(kamping::asserting_cast<size_t>(recv_buf_size));
            resized = true;
        }
    }

    bool resized = false;
};

struct resize_ext : std::ranges::range_adaptor_closure<resize_ext> {
    explicit resize_ext() = default;

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return resize_ext_view(std::forward<R>(r));
    }
};

template <typename T, typename Extent, typename LayoutPolicy>
requires std::same_as<LayoutPolicy, std::layout_left> || std::same_as<LayoutPolicy, std::layout_right>
struct mdspan_view {
    using mdspan = std::mdspan<T, Extent, LayoutPolicy>;
    mdspan base_;

    explicit mdspan_view(mdspan base) : base_(base), data_(base_.data_handle()) {}

    auto begin() noexcept {
        return data_;
    }
    auto end() noexcept {
        return data_ + base_.size();
    }
    auto data() {
        return data_;
    }

private:
    T* data_;
};

struct mdspan_view_fn {
    template <typename T, typename Extent, typename Layout>
    auto operator()(std::mdspan<T, Extent, Layout> ms) const {
        return mdspan_view<T, Extent, Layout>{ms};
    }
};

template <typename Mdspan>
auto operator|(Mdspan&& ms, mdspan_view_fn const& adaptor) {
    return adaptor(std::forward<Mdspan>(ms));
}

inline constexpr mdspan_view_fn mdspan_adapter{};