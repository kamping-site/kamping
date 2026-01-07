#pragma once

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"
#include "kamping/ranges/ranges.hpp"

using namespace kamping;

template <DataBufferConcept R, IntContiguousRange DisplsRange>
struct with_displs_view : pipe_view_interface<with_displs_view<R, DisplsRange>, R> {
    R base_;
    DisplsRange displs_;

  // FIXME: see below
    with_displs_view(R base, DisplsRange&& displs) : base_(std::move(base)), displs_(std::move(displs)) {}

    auto& displs() {
        return displs_;
    }
};

template<IntContiguousRange DisplsRange>
struct with_displs : std::ranges::range_adaptor_closure<with_displs<DisplsRange>> {
    DisplsRange displs_;

    explicit with_displs(DisplsRange&& displs) : displs_(std::forward<DisplsRange>(displs)) {}

    template <DataBufferConcept R>
    auto operator()(R&& r)  {
        return with_displs_view<std::ranges::views::all_t<R>, DisplsRange>(std::forward<R>(r), std::move(displs_));
    }
};

template<IntContiguousRange DisplsRange>
with_displs(DisplsRange&& displs) -> with_displs<std::views::all_t<DisplsRange>>;

template <BufferResizePolicy ResizePolicy, DataBufferConcept R, IntContiguousRange DisplsRange>
struct auto_displs_view : pipe_view_interface<auto_displs_view<ResizePolicy, R, DisplsRange>, R> {
    R           base_;
    DisplsRange displs_;
    bool        displs_set = false;

  // FIXME: This should be forward, and auto_displs_view should have a
  // deduction guide that dispatches to ranges::views::all_t, so we
  // can use this view without the pipe syntax.
    auto_displs_view(R base, DisplsRange&& displs) requires HasSizeV<R> : base_(std::move(base)),
                                                                          displs_(std::move(displs)) {}

    auto& displs() {
        if (!displs_set) {
            auto&  counts = base_.size_v();
            size_t ranks  = std::ranges::size(counts);
            kamping::ranges::resize<ResizePolicy>(displs_, ranks);
            KASSERT(
                std::ranges::size(displs_) >= ranks,
                "Displs are not large enough, and resize is not enabled",
                assert::light
            );

            std::exclusive_scan(
                std::ranges::begin(counts),
                std::ranges::begin(counts) + kamping::asserting_cast<int>(ranks),
                std::ranges::begin(displs_),
                0
            );
            displs_set = true;
        }
        return displs_;
    }
  
    void invalidate_displs() {
      displs_set = false;
    }
};


template <BufferResizePolicy ResizePolicy, IntContiguousRange DisplsRange>
struct auto_displs_adapter
    : std::ranges::range_adaptor_closure<auto_displs_adapter<ResizePolicy, DisplsRange>> {
    DisplsRange empty_displs_;

    auto_displs_adapter() : empty_displs_(DisplsRange()) {}
    explicit auto_displs_adapter(DisplsRange&& empty_displs) : empty_displs_(std::forward<DisplsRange>(empty_displs)) {}

    template <DataBufferConcept R>
    requires HasSizeV<R>
    auto operator()(R&& r) {
        return auto_displs_view<ResizePolicy, kamping::ranges::kamping_all_t<R>, DisplsRange>(
            std::forward<R>(r),
            std::move(empty_displs_)
        );
    }
};

template <BufferResizePolicy ResizePolicy = BufferResizePolicy::no_resize, IntContiguousRange DisplsRange>
auto auto_displs(DisplsRange&& empty_displs) {
    return auto_displs_adapter<ResizePolicy, kamping::ranges::kamping_all_t<DisplsRange>>(
        std::forward<DisplsRange>(empty_displs)
    );
}

template <IntContiguousRange DisplsRange = std::vector<int>>
auto auto_displs() {
  return auto_displs<BufferResizePolicy::resize_to_fit>(DisplsRange{});
}

