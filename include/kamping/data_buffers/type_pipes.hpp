#pragma once

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"
#include "kamping/ranges/ranges.hpp"

using namespace kamping;

template <DataBufferConcept R>
struct with_type_view : pipe_view_interface<with_type_view<R>, R> {
    R           base_;
    MPI_Datatype type_;

    with_type_view(R base, MPI_Datatype type) : base_(std::move(base)), type_(type) {}

    auto type() {
        return type_;
    }
};


struct with_type : std::ranges::range_adaptor_closure<with_type> {
    MPI_Datatype type_;

    explicit with_type(MPI_Datatype type) : type_(type) {}

    template <DataBufferConcept R>
    auto operator()(R&& r) {
        return with_type_view<kamping::ranges::kamping_all_t<R>>(std::forward<R>(r), type_);
    }
};