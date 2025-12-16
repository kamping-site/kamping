#pragma once

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"

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

template <BufferResizePolicy ResizePolicy, IntContiguousRange DisplsRange>
void resize_displs(DisplsRange& displs, size_t size) {
    if constexpr (ResizePolicy == BufferResizePolicy::no_resize) {
        return;
    }
    size_t const current_size = std::ranges::size(displs);
    if constexpr (ResizePolicy == BufferResizePolicy::grow_only) {
        if (current_size >= size) {
            return;
        }
    }
    if constexpr (ResizePolicy == BufferResizePolicy::resize_to_fit) {
        if (current_size == size) {
            return;
        }
    }
    displs.resize(size);
}

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
        // get ref to displs
        /* auto& displ_ref = displs_.base(); */
        if (!displs_set) {
            auto&  counts = base_.size_v();
            size_t ranks  = std::ranges::size(counts);
	    // FIXME: make this more generic: kamping::ranges::resize
            resize_displs<ResizePolicy>(displs_, ranks);
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

// displs_set needed because auto_displs() creates an empty DisplsRange which can only be resized after operator()
template <BufferResizePolicy ResizePolicy, IntContiguousRange DisplsRange/* , bool displs_set */>
struct auto_displs_adapter
    : std::ranges::range_adaptor_closure<auto_displs_adapter<ResizePolicy, DisplsRange/* , displs_set */>> {
    DisplsRange empty_displs_;

    auto_displs_adapter() : empty_displs_(DisplsRange()) {}
    explicit auto_displs_adapter(DisplsRange&& empty_displs) : empty_displs_(std::forward<DisplsRange>(empty_displs)) {}

    template <DataBufferConcept R>
    requires HasSizeV<R>
    auto operator()(R&& r) {
      // FIXME das kann weg, wir können einfach in auto_displs()
      // auto_displs(std::vector<int>{}) aufrufen, der Adapter kümmert
      // sich um resizing
        // if constexpr (!displs_set) {
        //     // size_v acts as ground truth for the number of ranks
        //     auto&  size_v = r.size_v();
        //     size_t ranks  = std::ranges::size(size_v);
        //     empty_displs_.base().resize(ranks);
        // }
        return auto_displs_view<ResizePolicy, std::ranges::views::all_t<R>, DisplsRange>(
            std::forward<R>(r),
            std::move(empty_displs_)
        );
    }
};

template <BufferResizePolicy ResizePolicy = BufferResizePolicy::no_resize, IntContiguousRange DisplsRange>
auto auto_displs(DisplsRange&& empty_displs) {
    return auto_displs_adapter<ResizePolicy, std::views::all_t<DisplsRange>/* , true */>(
        std::forward<DisplsRange>(empty_displs)
    );
}

template <IntContiguousRange DisplsRange = std::vector<int>>
auto auto_displs() {
  return auto_displs<BufferResizePolicy::resize_to_fit>(DisplsRange {});
}

