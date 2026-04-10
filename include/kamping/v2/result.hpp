#pragma once

#include <cstddef>
#include <utility>

namespace kamping::v2 {

/// Returned by blocking two-buffer operations (sendrecv, alltoall, etc.).
///
/// `SBuf` and `RBuf` are deduced from the factory's forwarding references:
///   - lvalue argument → template parameter is `T&`  → member is a reference (zero-copy borrow)
///   - rvalue argument → template parameter is `T`   → member is a value (ownership transfer)
///
/// Reference collapsing makes the two cases work with a single uniform member
/// declaration. Supports structured bindings: `auto [s, r] = blocking_op(...);`.
template <typename SBuf, typename RBuf>
struct result {
    SBuf send;
    RBuf recv;

    template <std::size_t I>
    decltype(auto) get() & {
        if constexpr (I == 0) return (send);
        else return (recv);
    }

    template <std::size_t I>
    decltype(auto) get() && {
        if constexpr (I == 0) return std::forward<SBuf>(send);
        else return std::forward<RBuf>(recv);
    }

    template <std::size_t I>
    decltype(auto) get() const& {
        if constexpr (I == 0) return (send);
        else return (recv);
    }
};

// Free get<I> overloads in kamping::v2 for ADL lookup during structured bindings.
template <std::size_t I, typename SBuf, typename RBuf>
decltype(auto) get(result<SBuf, RBuf>& r) {
    return r.template get<I>();
}

template <std::size_t I, typename SBuf, typename RBuf>
decltype(auto) get(result<SBuf, RBuf>&& r) {
    return std::move(r).template get<I>();
}

template <std::size_t I, typename SBuf, typename RBuf>
decltype(auto) get(result<SBuf, RBuf> const& r) {
    return r.template get<I>();
}

} // namespace kamping::v2

// tuple_size / tuple_element specializations required for structured bindings.
namespace std {

template <typename SBuf, typename RBuf>
struct tuple_size<kamping::v2::result<SBuf, RBuf>> : integral_constant<size_t, 2> {};

template <typename SBuf, typename RBuf>
struct tuple_element<0, kamping::v2::result<SBuf, RBuf>> {
    using type = SBuf;
};

template <typename SBuf, typename RBuf>
struct tuple_element<1, kamping::v2::result<SBuf, RBuf>> {
    using type = RBuf;
};

} // namespace std
