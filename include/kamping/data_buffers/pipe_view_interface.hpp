#pragma once

#include <ranges>

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/data_buffers/data_buffer_concepts.hpp"

template <typename Derived, typename Base>
class pipe_view_interface : public std::ranges::view_interface<Derived> {
public:
    constexpr auto& base() noexcept {
        return static_cast<Derived&>(*this).base_;
    }

    constexpr auto get_base() noexcept {
        auto& b = base();
        if constexpr (requires { b.get_base(); }) {
            return b.get_base();
        } else {
            return b;
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

    constexpr void set_size(std::size_t n) requires kamping::HasSetSize<Base> {
        base().set_size(n);
    }

    constexpr auto type() requires kamping::HasType<Base> {
        return base().type();
    }

    constexpr void set_size_v(std::vector<int>&& size_v) requires kamping::HasSetSizeV<Base> {
        return base().set_size_v(std::move(size_v));
    }

    constexpr void set_displs(std::vector<int>&& displs) requires kamping::HasSetDispls<Base> {
        return base().set_displs(std::move(displs));
    }
};