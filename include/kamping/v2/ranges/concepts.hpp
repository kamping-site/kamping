#pragma once

#include <concepts>

#include <mpi.h>

#include "ranges.hpp"

namespace kamping::ranges {

template <typename T>
concept has_mpi_size = requires(T const& t) {
    { kamping::ranges::size(t) } -> integer_like;
};

template <typename T>
concept has_mpi_data = requires(T&& t) {
    { kamping::ranges::data(t) } -> ptr_to_object;
};

template <typename T>
concept has_mpi_type = requires(T const& t) {
    { kamping::ranges::type(t) } -> std::convertible_to<MPI_Datatype>;
};

template <typename T>
concept data_buffer = has_mpi_size<T> && has_mpi_data<T> && has_mpi_type<T>;

template <typename T>
concept send_buffer = data_buffer<T> && requires(T&& t) {
    { kamping::ranges::data(t) } -> std::convertible_to<void const*>;
};

template <typename T>
concept recv_buffer = data_buffer<T> && requires(T&& t) {
    { kamping::ranges::data(t) } -> std::convertible_to<void*>;
};

/// Recv buffer whose size is not known upfront. The collective calls set_recv_count(n) after
/// inferring n; the view then lazily resizes on first mpi_data() access.
template <typename T>
concept resizable_recv_buf = requires(T& t, std::ptrdiff_t n) {
    t.set_recv_count(n);
};

// ──────────────────────────────────────────────────────────────────────────────
// enable_borrowed_buffer — opt-in trait for non-owning buffer types.
//
// A "borrowed" buffer does not own its data: the underlying memory is managed
// externally and will outlive the buffer object. This mirrors std::ranges::
// enable_borrowed_range. The default covers any std::ranges::borrowed_range
// (e.g. std::span, std::string_view). Kamping views specialize this in their
// own headers to propagate borrowedness from their Base type.
//
// To opt in a custom non-owning buffer type:
//   template <>
//   inline constexpr bool kamping::ranges::enable_borrowed_buffer<MyView> = true;
// ──────────────────────────────────────────────────────────────────────────────
template <typename T>
inline constexpr bool enable_borrowed_buffer = std::ranges::borrowed_range<T>;

template <typename T>
concept borrowed_buffer = enable_borrowed_buffer<std::remove_cvref_t<T>>;

// Result type for single-buffer returns (send, recv):
//   - non-borrowed rvalue: by value (ownership transfer)
//   - lvalue reference or borrowed: passthrough reference
template <typename Buf>
using buf_result_t = std::conditional_t<
    !std::is_reference_v<Buf> && !borrowed_buffer<Buf>,
    std::remove_cvref_t<Buf>,
    Buf&&>;

// Result type for tuple elements (sendrecv, collectives):
//   - lvalue reference: kept as reference
//   - rvalue (owned or borrowed): by value — avoids T&& in tuples
template <typename Buf>
using buf_tuple_element_t =
    std::conditional_t<std::is_lvalue_reference_v<Buf>, Buf, std::remove_cvref_t<Buf>>;

} // namespace kamping::ranges
