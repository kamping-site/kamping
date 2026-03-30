#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "kamping/v2/ranges/adaptor_closure.hpp"

namespace kamping::ranges {

/// A closure with pre-bound arguments. Created by calling an adaptor with only the extra arguments
/// (partial application). When invoked (directly or via |), prepends the incoming value and calls fn_.
///
/// Example: with_type(MPI_INT) returns a bound_adaptor holding fn_ and MPI_INT in bound_.
///          vec | that_closure  →  fn_(vec, MPI_INT)  →  with_type_view(vec, MPI_INT)
template <typename Fn, typename... BoundArgs>
struct bound_adaptor : adaptor_closure<bound_adaptor<Fn, BoundArgs...>> {
    [[no_unique_address]] Fn fn_;
    std::tuple<BoundArgs...>  bound_;

    constexpr bound_adaptor(Fn fn, std::tuple<BoundArgs...> bound)
        : fn_(std::move(fn)),
          bound_(std::move(bound)) {}

    template <typename T>
    constexpr auto operator()(T&& val) const {
        return std::apply(
            [&](auto const&... args) { return fn_(std::forward<T>(val), args...); },
            bound_
        );
    }
};

/// Generic range adaptor factory, parameterized by the number of extra arguments (beyond the value).
/// Arity-based disambiguation (like libstdc++ internally):
///   - ExtraArgs arguments       → partial application, returns a pipeable bound_adaptor
///   - ExtraArgs + 1 arguments   → full call, first argument is the value
///
/// Usage:  inline constexpr adaptor<1, decltype([](auto&& r, MPI_Datatype t) { ... })> with_type{};
///         vec | with_type(MPI_INT)       // partial → bound_adaptor → pipe applies it
///         with_type(vec, MPI_INT)        // full call
template <std::size_t ExtraArgs, typename Fn>
struct adaptor {
    [[no_unique_address]] Fn fn_;

    /// Partial application: bind ExtraArgs arguments, return a pipeable closure.
    template <typename... Args>
        requires(sizeof...(Args) == ExtraArgs)
    constexpr auto operator()(Args&&... args) const {
        return bound_adaptor<Fn, std::decay_t<Args>...>(
            fn_, std::tuple{std::forward<Args>(args)...}
        );
    }

    /// Full call: first argument is the value, remaining ExtraArgs are forwarded to fn_.
    template <typename T, typename... Args>
        requires(sizeof...(Args) == ExtraArgs)
    constexpr auto operator()(T&& val, Args&&... args) const {
        return fn_(std::forward<T>(val), std::forward<Args>(args)...);
    }
};

} // namespace kamping::ranges
