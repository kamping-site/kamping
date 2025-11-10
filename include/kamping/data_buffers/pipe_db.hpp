#pragma once

#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"

template <std::ranges::contiguous_range R>
struct auto_displs_view : pipe_view_interface<auto_displs_view<R>, R> {
    R    base_;
    bool displs_set = false;

    explicit auto_displs_view(R base) requires kamping::HasDispls<R> && kamping::HasSetDispls<R> && kamping::HasSizeV<R>
        : base_(std::move(base)) {}

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
    R    base_;
    bool resized = false;

    explicit resize_ext_view(R base) requires kamping::HasDispls<R> && kamping::HasSetSize<R> && kamping::HasSizeV<R>
        : base_(std::move(base)) {}

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
};

struct resize_ext : std::ranges::range_adaptor_closure<resize_ext> {
    explicit resize_ext() = default;

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return resize_ext_view(std::forward<R>(r));
    }
};

template <std::ranges::contiguous_range R>
struct displs_view : pipe_view_interface<resize_ext_view<R>, R> {
    R                base_;
    std::vector<int> displs_;

    explicit displs_view(R&& base) : base_(std::move(base)) {}
    displs_view(R&& base, std::vector<int> displs) : base_(std::move(base)), displs_(std::move(displs)) {}

    auto displs() {
        return displs_;
    }

    void set_displs(std::vector<int>&& displs) {
        displs_ = displs;
    }
};

template <typename R>
displs_view(R&&, std::vector<int> displs) -> displs_view<std::ranges::views::all_t<R>>;

template <typename R>
displs_view(R&&) -> displs_view<std::ranges::views::all_t<R>>;

struct add_displs : std::ranges::range_adaptor_closure<add_displs> {
    std::vector<int> displs_;
    bool             displs_set_ = false;

    explicit add_displs(std::vector<int> displs) : displs_(std::move(displs)), displs_set_(true) {}
    add_displs() = default;

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return displs_set_ ? displs_view(std::forward<R>(r), displs_) : displs_view(std::forward<R>(r));
    }
};

template <std::ranges::contiguous_range R>
struct size_v_view : pipe_view_interface<size_v_view<R>, R> {
    R                base_;
    std::vector<int> size_v_;

    size_v_view(R&& base, std::vector<int> size_v) : base_(std::move(base)), size_v_(std::move(size_v)) {}

    auto size_v() {
        return size_v_;
    }
};

template <typename R>
size_v_view(R&&, std::vector<int> size_v) -> size_v_view<std::ranges::views::all_t<R>>;

struct add_size_v : std::ranges::range_adaptor_closure<add_size_v> {
    std::vector<int> size_v_;

    explicit add_size_v(std::vector<int> size_v) : size_v_(std::move(size_v)) {}

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return size_v_view(std::forward<R>(r), size_v_);
    }
};
