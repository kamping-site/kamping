#pragma once
#include "kamping/data_buffer.hpp"
#include "kamping/data_buffers/data_buffer_concepts.hpp"

namespace kamping::ranges {
template <BufferResizePolicy ResizePolicy, IntContiguousRange IntRange>
void resize(IntRange& displs, size_t size) {
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

template <std::ranges::range Range>
class kamping_owning_view : public std::ranges::owning_view<Range> {
public:
    kamping_owning_view(Range&& r) : std::ranges::owning_view<Range>{std::forward<Range>(r)} {}

    void resize(size_t size) requires requires {
        std::declval<Range>().resize(size);
    }
    { this->base().resize(size); }

    auto& displs() requires requires {
        std::declval<Range>().displs();
    }
    { return this->base().displs(); }

    auto& size_v() requires requires {
        std::declval<Range>().size_v();
    }
    { return this->base().size_v(); }
};

template <std::ranges::range Range>
class kamping_ref_view : public std::ranges::ref_view<Range> {
public:
    kamping_ref_view(Range& r) : std::ranges::ref_view<Range>{r} {}

    void resize(size_t size) requires requires {
        std::declval<Range>().resize(size);
    }
    { this->base().resize(size); }

    auto& displs() requires requires {
        std::declval<Range>().displs();
    }
    { return this->base().displs(); }

    auto& size_v() requires requires {
        std::declval<Range>().size_v();
    }
    { return this->base().size_v(); }
};

template <typename Range>
concept can_ref_view = requires {
    std::ranges::ref_view{std::declval<Range>()};
};
template <typename Range>
concept can_owning_view = requires {
    std::ranges::owning_view{std::declval<Range>()};
};

template <std::ranges::viewable_range Range>
requires std::ranges::view<std::decay_t<Range>> || can_ref_view<Range> || can_owning_view<Range>
auto kamping_all(Range&& r) {
    using range_type = std::remove_cvref_t<Range>;

    if constexpr (std::ranges::view<std::decay_t<Range>>)
        return std::forward<Range>(r);
    else if constexpr (can_ref_view<Range>)
        return kamping_ref_view<range_type>{std::forward<Range>(r)};
    else
        return kamping_owning_view<range_type>{std::forward<Range>(r)};
}

template <std::ranges::viewable_range Range>
using kamping_all_t = decltype(kamping_all(std::declval<Range>()));

} // namespace kamping::ranges