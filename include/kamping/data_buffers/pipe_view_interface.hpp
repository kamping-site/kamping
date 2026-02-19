#pragma once

#include <ranges>

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/data_buffers/data_buffer_concepts.hpp"

template <typename Derived, typename Base>
class pipe_view_interface : public std::ranges::view_interface<Derived> {
public:

    using infer_tag = Base::infer_tag;

    constexpr auto& base() & noexcept {
        return static_cast<Derived&>(*this).base_;
    }

    constexpr auto&& base() && noexcept {
        return std::move(static_cast<Derived&>(*this).base_);
    }

    constexpr auto& buffer() noexcept {
        auto& b = base();
        if constexpr (requires { b.buffer(); }) {
            return b.buffer();
        } else {
            return b;
        }
    }

    constexpr auto extract_buffer() noexcept {
        auto&& b = base();
        if constexpr (requires { b.extract_buffer(); }) {
            return std::move(b).extract_buffer();
        } else {
            return std::move(b);
        }
    }

    constexpr auto begin() {
        return std::ranges::begin(base());
    }

    constexpr auto end() {
        return std::ranges::end(base());
    }

    constexpr auto data() {
        return std::ranges::data(base());
    }

    constexpr auto& displs() requires kamping::HasDispls<Base> {
        return base().displs();
    }

    constexpr auto& size_v() requires kamping::HasSizeV<Base> {
        return base().size_v();
    }

    constexpr auto type() requires kamping::HasType<Base> {
        return base().type();
    }

    constexpr void resize(std::size_t n) requires kamping::HasResize<Base>
    { base().resize(n); }
};