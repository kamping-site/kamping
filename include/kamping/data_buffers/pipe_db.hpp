#pragma once

#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"



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



