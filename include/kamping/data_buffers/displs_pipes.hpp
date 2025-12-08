#pragma once

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"

using namespace kamping;
// with displs -> store given displs. either value, ref or move
template <std::ranges::contiguous_range R>
struct with_displs_view : pipe_view_interface<resize_ext_view<R>, R> {
    R                base_;
    std::vector<int> displs_;

    explicit with_displs_view(R&& base) : base_(std::move(base)) {}
    with_displs_view(R&& base, std::vector<int> displs) : base_(std::move(base)), displs_(std::move(displs)) {}

    auto& displs() {
        return displs_;
    }

    void set_displs(std::vector<int>&& displs) {
        displs_ = displs;
    }
};

template <typename R>
with_displs_view(R&&, std::vector<int> displs) -> with_displs_view<std::ranges::views::all_t<R>>;

template <typename R>
with_displs_view(R&&) -> with_displs_view<std::ranges::views::all_t<R>>;

struct with_displs : std::ranges::range_adaptor_closure<with_displs> {
    std::vector<int> displs_;
    bool             displs_set_ = false;

    explicit with_displs(std::vector<int> displs) : displs_(std::move(displs)), displs_set_(true) {}
    with_displs() = default;

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return displs_set_ ? displs_view(std::forward<R>(r), displs_) : displs_view(std::forward<R>(r));
    }
};


// Compute displs, called without arguments -> construct displs vector, or called with empty range
//TODO : change template order to let user decide on DisplsRange type
template <BufferResizePolicy ResizePolicy, std::ranges::contiguous_range R, IntContiguousRange DisplsRange = std::vector<int>>
struct auto_displs_view : pipe_view_interface<auto_displs_view<ResizePolicy, R, DisplsRange>, R> {

    R    base_;
    std::variant<std::ranges::ref_view<DisplsRange>,
                 std::ranges::owning_view<DisplsRange>> displs_;
    bool displs_set = false;

    // If no empty_displs given, construct DisplsRange of correct size
    explicit auto_displs_view(R base) requires HasSizeV<R>
        : base_(std::forward<R>(base)), displs_(std::ranges::owning_view<DisplsRange>(DisplsRange(std::ranges::size(base_.size_v())))) {}


    auto_displs_view(R base, std::variant<std::ranges::ref_view<DisplsRange>, std::ranges::owning_view<DisplsRange>> displs)
    : base_(std::forward<R>(base)), displs_(std::move(displs)) {}

    auto& displs() {
        // get ref to displs
        auto& displ_ref = std::visit([](auto& view) -> DisplsRange& {return view.base(); }, displs_);
        if (!displs_set) {

            auto   counts = base_.size_v();
            size_t ranks  = std::ranges::size(counts);
            if (ResizePolicy != BufferResizePolicy::no_resize) {
                displ_ref.resize(ranks);
            }
            KASSERT(std::ranges::size(displ_ref) >= ranks, "Displs are not large enough, and resize is not enabled", assert::light);

            std::exclusive_scan(
                counts.begin(),
                counts.begin() + kamping::asserting_cast<int>(ranks),
                displ_ref.begin(),
                0
            );
            displs_set = true;
        }
        return displ_ref;
    }
};

template <BufferResizePolicy ResizePolicy = BufferResizePolicy::no_resize, IntContiguousRange DisplsRange = std::vector<int>>
struct auto_displs : std::ranges::range_adaptor_closure<auto_displs<ResizePolicy, DisplsRange>> {
    std::variant<std::ranges::ref_view<DisplsRange>,
    std::ranges::owning_view<DisplsRange>> empty_displs_{
        std::ranges::owning_view<DisplsRange>{}
    };
    bool data_given_ = false;

    auto_displs() = default;

    explicit auto_displs(DisplsRange&& empty_displs) : empty_displs_(std::ranges::owning_view<DisplsRange>(std::move(empty_displs))), data_given_(true) {}
    explicit auto_displs(DisplsRange& empty_displs) : empty_displs_(std::ranges::ref_view<DisplsRange>(empty_displs)), data_given_(true) {}

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r)  {
        if (data_given_) {
            return auto_displs_view<ResizePolicy, std::ranges::views::all_t<R>, DisplsRange>(std::forward<R>(r), std::move(empty_displs_));
        }
        else {
            return auto_displs_view<ResizePolicy, std::ranges::views::all_t<R>, DisplsRange>(std::forward<R>(r));
        }
    }
};


