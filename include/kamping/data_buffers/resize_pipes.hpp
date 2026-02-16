#pragma once

#include <ranges>

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"

using namespace kamping;

template <DataBufferConcept R>
struct resize_vbuf_view : pipe_view_interface<resize_vbuf_view<R>, R> {
    R    base_;
    bool resized = false;

    explicit resize_vbuf_view(R base) requires HasDispls<R> && HasSizeV<R> : base_(std::forward<R>(base)) {}

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
            auto& displs = this->displs();
            auto& counts = this->size_v();

            auto counts_ptr = std::ranges::data(displs);
            auto displs_ptr = std::ranges::data(counts);

            size_t ranks = std::ranges::size(counts);

            int recv_buf_size = 0;
            for (size_t i = 0; i < ranks; ++i) {
                recv_buf_size = std::max(recv_buf_size, *(counts_ptr + i) + *(displs_ptr + i));
            }

            base_.resize(kamping::asserting_cast<size_t>(recv_buf_size));
            resized = true;
        }
    }
};

struct resize_vbuf : std::ranges::range_adaptor_closure<resize_vbuf> {
    explicit resize_vbuf() = default;

    template <DataBufferConcept R>
    requires HasDispls<R> && HasSizeV<R> && HasResize<R>
    auto operator()(R&& r) const {
        return resize_vbuf_view(std::forward<R>(r));
    }
};
