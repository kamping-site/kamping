#pragma once
#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"
#include "kamping/ranges/ranges.hpp"

using namespace kamping;

template <DataBufferConcept R, IntContiguousRange SizeRange>
struct with_size_v_view : pipe_view_interface<with_size_v_view<R, SizeRange>, R> {
    R         base_;
    SizeRange size_v_;

    with_size_v_view(R base, SizeRange size_v) : base_(std::move(base)), size_v_(std::move(size_v)) {}

    auto& size_v() {
        return size_v_;
    }
};

template <DataBufferConcept R, IntContiguousRange SizeRange>
auto make_size_v_view(R&& base, SizeRange&& size_v) {
    return with_size_v_view<kamping::ranges::kamping_all_t<R>, kamping::ranges::kamping_all_t<SizeRange>>(
        std::forward<R>(base),
        std::forward<SizeRange>(size_v)
    );
}

template <IntContiguousRange SizeRange>
struct with_size_v : std::ranges::range_adaptor_closure<with_size_v<SizeRange>> {
    SizeRange size_v_;

    explicit with_size_v(SizeRange&& size_v) : size_v_(std::forward<SizeRange>(size_v)) {}

    template <DataBufferConcept R>
    auto operator()(R&& r) {
        return with_size_v_view<kamping::ranges::kamping_all_t<R>, SizeRange>(std::forward<R>(r), std::move(size_v_));
    }
};

template <IntContiguousRange SizeRange>
with_size_v(SizeRange&& size_v) -> with_size_v<kamping::ranges::kamping_all_t<SizeRange>>;