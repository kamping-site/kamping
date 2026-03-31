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
} // namespace kamping::ranges
