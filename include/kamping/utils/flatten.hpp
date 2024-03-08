// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once
#include <numeric>
#include <utility>
#include <vector>

#include <kamping/utils/traits.hpp>

#include "kamping/named_parameters.hpp"

namespace kamping {
namespace internal {

/// @brief A wrapper around a functor \p F that makes it callable using the `call` method.
template <typename F>
struct CallableWrapper {
    F f; ///< The functor to wrap.

    /// @brief Calls the wrapped functor with the given arguments.
    template <typename... Args>
    auto call(Args&&... args) {
        return f(std::forward<Args...>(args)...);
    }
};

/// @brief A factory function for \ref CallableWrapper.
template <typename F>
auto make_callable_wrapper(F f) {
    return CallableWrapper<F>{std::move(f)};
}

/// @brief Maps a container to is underlying nested container.
template <typename T, typename Enable = void>
struct FlatContainer {};

/// @brief Maps a container to is underlying nested container.
template <typename T>
struct FlatContainer<T, std::enable_if_t<is_sparse_send_buffer_v<T>>> {
    using type =
        std::remove_const_t<std::tuple_element_t<1, typename T::value_type>>; ///< The type of the nested container.
};

/// @brief Maps a container to is underlying nested container.
template <typename T>
struct FlatContainer<T, std::enable_if_t<is_nested_send_buffer_v<T>>> {
    using type = std::remove_const_t<typename T::value_type>; ///< The type of the nested container.
};

} // namespace internal

/// @brief Flattens a container of containers or destination-container-pairs and provides the flattened buffer, send
/// counts and send displacements as parameters to be passed to an \c MPI call.
///
/// This returns a callable wrapper that can be called with a functor which accepts a parameter pack of these arguments.
///
/// Example:
/// ```cpp
/// Communicator                  comm;
/// std::vector<std::vector<int>> nested_send_buf(comm.size()); // or std::unordered_map<int, std::vector<int>>
/// auto [recv_buf, recv_counts, recv_displs] = with_flattened(nested_send_buf).call([&](auto... flattened) {
///    return comm.alltoallv(std::move(flattened)..., recv_counts_out(), recv_displs_out());
/// });
/// ```
///
/// The container can be a range of pair-like types of destination and data (see \c is_sparse_send_buffer_v ) or
/// a container of containers (see \c is_nested_send_buffer_v ).
///
/// @param nested_send_buf The nested container of send buffers.
/// @param comm_size The size of the communicator, used as number of elements in the computed count buffers.
/// @tparam CountContainer The type of the container to use for the send counts and send displacements.
/// @tparam Container The type of the nested container.
/// @tparam Enable SFINAE.
///
template <
    template <typename...> typename CountContainer = std::vector,
    typename Container,
    typename Enable = std::enable_if_t<is_sparse_send_buffer_v<Container> || is_nested_send_buffer_v<Container>>>
auto with_flattened(Container const& nested_send_buf, size_t comm_size) {
    CountContainer<int> send_counts(comm_size);
    if constexpr (is_sparse_send_buffer_v<Container>) {
        for (auto const& [destination, message]: nested_send_buf) {
            auto send_buf                                    = kamping::send_buf(message);
            send_counts[asserting_cast<size_t>(destination)] = asserting_cast<int>(send_buf.size());
        }
    } else {
        static_assert(is_nested_send_buffer_v<Container>);
        size_t i = 0;
        for (auto const& message: nested_send_buf) {
            auto send_buf  = kamping::send_buf(message);
            send_counts[i] = asserting_cast<int>(send_buf.size());
            i++;
        }
    }
    CountContainer<int> send_displs(comm_size);
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
    size_t total_send_count = asserting_cast<size_t>(send_displs.back() + send_counts.back());
    typename internal::FlatContainer<Container>::type flat_send_buf;
    flat_send_buf.resize(total_send_count);
    if constexpr (is_sparse_send_buffer_v<Container>) {
        for (auto const& [destination, message]: nested_send_buf) {
            auto send_buf = kamping::send_buf(message).construct_buffer_or_rebind();
            std::copy_n(
                send_buf.data(),
                send_buf.size(),
                flat_send_buf.data() + send_displs[asserting_cast<size_t>(destination)]
            );
        }
    } else {
        static_assert(is_nested_send_buffer_v<Container>);
        size_t i = 0;
        for (auto const& message: nested_send_buf) {
            auto send_buf = kamping::send_buf(message).construct_buffer_or_rebind();
            std::copy_n(send_buf.data(), send_buf.size(), flat_send_buf.data() + send_displs[i]);
            i++;
        }
    }
    return internal::make_callable_wrapper([flat_send_buf = std::move(flat_send_buf),
                                            send_counts   = std::move(send_counts),
                                            send_displs   = std::move(send_displs)](auto&& f) {
        return std::apply(
            std::forward<decltype(f)>(f),
            std::tuple(
                kamping::send_buf(flat_send_buf),
                kamping::send_counts(send_counts),
                kamping::send_displs(send_displs)
            )
        );
    });
}

/// @brief Flattens a container of containers and provides the flattened buffer, send counts and send
/// displacements as parameters to be passed to an \c MPI call.
///
/// This returns a callable wrapper that can be called with a functor which accepts a parameter pack of these arguments.
/// The size of the computed count buffers is the size of the container.
///
/// Example:
/// ```cpp
/// Communicator                  comm;
/// std::vector<std::vector<int>> nested_send_buf(comm.size());
/// auto [recv_buf, recv_counts, recv_displs] = with_flattened(nested_send_buf).call([&](auto... flattened) {
///   return comm.alltoallv(std::move(flattened)..., recv_counts_out(), recv_displs_out());
/// });
/// ```
///
/// @param nested_send_buf The nested container of send buffers. Must satisfy \c is_nested_send_buffer_v.
/// @tparam CountContainer The type of the container to use for the send counts and send displacements.
/// @tparam Container The type of the nested container.
/// @tparam Enable SFINAE.
///
template <
    template <typename...> typename CountContainer = std::vector,
    typename Container,
    typename Enable = std::enable_if_t<is_nested_send_buffer_v<Container>>>
auto with_flattened(Container const& nested_send_buf) {
    return with_flattened<CountContainer>(nested_send_buf, nested_send_buf.size());
}
} // namespace kamping
