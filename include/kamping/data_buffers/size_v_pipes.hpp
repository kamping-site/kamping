#pragma once
#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"


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