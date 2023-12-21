// This file is part of KaMPIng.
//
// Copyright 2021-2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.:

#pragma once

/// @file
/// @brief Some functions and types simplifying/enabling the development of wrapped \c MPI calls in KaMPIng.

#include <iostream>
#include <optional>
#include <tuple>
#include <utility>

#include "kamping/has_member.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/named_parameters.hpp"
#include "named_parameter_selection.hpp"

template <typename>
class TD;

namespace kamping {
template <typename... Args>
class MPIResult_ {
public:
    MPIResult_(std::tuple<Args...> data) : _data(std::move(data)) {}

    /// @brief Extracts the \c kamping::Status from the MPIResult object.
    ///
    /// This function is only available if the underlying status is owned by the
    /// MPIResult object.
    /// @tparam StatusType_ Template parameter helper only needed to remove this
    /// function if StatusType does not possess a member function \c extract().
    /// @return Returns the underlying status object.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_tuple<internal::ParameterType::status, T>(), bool> = true>
    decltype(auto) extract_status() {
        return internal::select_parameter_type_tuple<internal::ParameterType::status>(_data).extract();
    }

    /// @brief Extracts the \c recv_buffer from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the
    /// MPIResult object.
    /// @tparam RecvBuf_ Template parameter helper only needed to remove this
    /// function if RecvBuf does not possess a member function \c extract().
    /// @return Returns the underlying storage containing the received elements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_tuple<internal::ParameterType::recv_buf, T>(), bool> = true>
    decltype(auto) extract_recv_buffer() {
        return internal::select_parameter_type_tuple<internal::ParameterType::recv_buf>(_data).extract();
    }

    /// @brief Extracts the \c recv_counts from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvCounts_ Template parameter helper only needed to remove this function if RecvCounts does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the receive counts.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_tuple<internal::ParameterType::recv_counts, T>(), bool> = true>
    decltype(auto) extract_recv_counts() {
        return internal::select_parameter_type_tuple<internal::ParameterType::recv_counts>(_data).extract();
    }

    /// @brief Extracts the \c recv_count from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvCount_ Template parameter helper only needed to remove this function if RecvCount does not
    /// possess a member function \c extract().
    /// @return Returns the underlying storage containing the recv count.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_tuple<internal::ParameterType::recv_count, T>(), bool> = true>
    decltype(auto) extract_recv_count() {
        return internal::select_parameter_type_tuple<internal::ParameterType::recv_count>(_data).extract();
    }

    /// @brief Extracts the \c recv_displs from the MPIResult object.
    ///
    /// This function is only available if the underlying memory is owned by the MPIResult object.
    /// @tparam RecvDispls_ Template parameter helper only needed to remove this function if RecvDispls does not possess
    /// a member function \c extract().
    /// @return Returns the underlying storage containing the receive displacements.
    template <
        typename T = std::tuple<Args...>,
        std::enable_if_t<internal::has_parameter_tuple<internal::ParameterType::recv_displs, T>(), bool> = true>
    decltype(auto) extract_recv_displs() {
        return internal::select_parameter_type_tuple<internal::ParameterType::recv_displs>(_data).extract();
    }

    // structured bindings getter
    template <std::size_t i>
    auto get() {
        return std::get<i>(_data).extract();
    }

    std::tuple<Args...> _data;
};

} // namespace kamping

namespace std {
template <typename... Args>
struct tuple_size<kamping::MPIResult_<Args...>> {
    static constexpr size_t value = sizeof...(Args);
};

template <size_t index, typename... Args>
struct tuple_element<index, kamping::MPIResult_<Args...>> {
    using type = decltype(declval<kamping::MPIResult_<Args...>>().template get<index>());
};
} // namespace std

namespace kamping {
namespace internal {
// based on https://stackoverflow.com/a/18366475
template <typename, typename>
struct Concat {};

template <typename First, typename... Remainder>
struct Concat<First, std::tuple<Remainder...>> {
    using type = std::tuple<First, Remainder...>;
};

template <ParameterType ptype_>
struct Item {
    static constexpr ParameterType ptype = ptype_;
};

template <typename...>
struct Filter;
template <>
struct Filter<> {
    using type = std::tuple<>;
};

template <typename First, typename... Remainder>
struct Filter<First, Remainder...> {
    using non_ref_first             = std::remove_reference_t<First>;
    static constexpr bool predicate = non_ref_first::is_owning && non_ref_first::is_out_buffer
                                      && non_ref_first::parameter_type != ParameterType::recv_buf;
    static constexpr ParameterType ptype = non_ref_first::parameter_type;
    using type                           = std::conditional_t<
        predicate,
        typename Concat<Item<ptype>, typename Filter<Remainder...>::type>::type,
        typename Filter<Remainder...>::type>;
};

template <typename First>
struct PrependRecvBufferAndFilter {
    using type = typename Concat<Item<ParameterType::recv_buf>, First>::type;
};

template <ParameterType ptype, typename ForwardTuple>
auto& retrieve_element(ForwardTuple&& forward_tuple) {
    return select_parameter_type_tuple<ptype>(forward_tuple);
}

template <typename outparameters, typename forwardtuple, std::size_t... i>
auto construct_tuple_impl(forwardtuple&& forward_tuple, std::index_sequence<i...>) {
    return std::make_tuple(std::move(retrieve_element<std::tuple_element_t<i, outparameters>::ptype>(forward_tuple))...
    );
}

template <typename outparameters, typename forwardtuple>
auto construct_tuple(forwardtuple&& forward_tuple) {
    constexpr std::size_t num_output_parameters = std::tuple_size_v<outparameters>;
    return construct_tuple_impl<outparameters>(
        std::forward<forwardtuple>(forward_tuple),
        std::make_index_sequence<num_output_parameters>{}
    );
}

template <typename OutputParameters, typename... Args>
auto make_result(Args&&... args) {
    static_assert(internal::has_parameter_type<internal::ParameterType::recv_buf, Args...>());
    auto&          recv_buffer        = internal::select_parameter_type<internal::ParameterType::recv_buf>(args...);
    constexpr bool recv_buf_is_owning = std::remove_reference_t<decltype(recv_buffer)>::is_owning;
    // constexpr bool output_params_contain_recv_buf = has_parameter_type<ParameterType::recv_buf, OutputParameters>;
    if constexpr (recv_buf_is_owning) {
        using type = typename PrependRecvBufferAndFilter<OutputParameters>::type;
        // TD<OutputParameters> td1;
        // TD<type> td2;
        if constexpr (std::tuple_size_v<type> == 1) {
            return recv_buffer.extract();
        } else {
            return MPIResult_(construct_tuple<type>(std::forward_as_tuple(args...)));
        }
    } else {
        using type = typename Filter<OutputParameters>::type;
        return MPIResult_(construct_tuple<type>(std::forward_as_tuple(args...)));
    }

    // if OutputParameters has only recv_buffer
    //  return recv buffer directly;
    // else if OutputParameters contains recv buffer and others
    //  reorder parameters such that recv buffer comes first
    // else
    //  do not reorder arguments
}

} // namespace internal
} // namespace kamping
