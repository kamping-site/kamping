// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
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

#include <tuple>
#include <type_traits>

#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_type_definitions.hpp"

namespace kamping::internal {

template <typename ParametersTuple, typename... Args>
struct has_all_required_parameters {
    template <size_t... Indices>
    static constexpr auto number_of_required(std::index_sequence<Indices...>) {
        return std::tuple_size_v<decltype(std::tuple_cat(
            std::conditional_t<
                has_parameter_type<std::tuple_element_t<Indices, ParametersTuple>::value, Args...>(),
                std::tuple<std::tuple_element_t<Indices, ParametersTuple>>, std::tuple<>>{}...))>;
    }

    static constexpr bool assertion =
        (std::tuple_size_v<
             ParametersTuple> == number_of_required(std::make_index_sequence<std::tuple_size_v<ParametersTuple>>{}));
}; // struct has_all_required_parameters

template <typename RequiredParametersTuple, typename OptionalParametersTuple, typename... Args>
struct has_no_unused_parameters {
    using all_available_parameters = decltype(std::tuple_cat(RequiredParametersTuple{}, OptionalParametersTuple{}));

    template <size_t... Indices>
    static constexpr auto total_number_of_parameter(std::index_sequence<Indices...>) {
        return std::tuple_size_v<decltype(std::tuple_cat(
                   std::conditional_t<
                       !has_parameter_type<std::tuple_element_t<Indices, all_available_parameters>::value, Args...>(),
                       std::tuple<std::tuple_element_t<Indices, all_available_parameters>>,
                       std::tuple<>>{}...))> + sizeof...(Args);
    }

    static constexpr bool assertion =
        (std::tuple_size_v<all_available_parameters> >= total_number_of_parameter(
             std::make_index_sequence<std::tuple_size_v<all_available_parameters>>{}));

}; // struct has_no_unused_parameters

template <typename Tuple>
struct all_unique : std::true_type {};

template <typename T, typename... Ts>
struct all_unique<std::tuple<T, Ts...>>
    : std::conjunction<std::negation<std::disjunction<std::is_same<T, Ts>...>>, all_unique<std::tuple<Ts...>>> {};

template <typename Tuple>
static constexpr bool all_unique_v = all_unique<Tuple>::value;

template <ParameterType parameter_type>
struct parameter_types_to_integral_constant {
    using type = std::integral_constant<ParameterType, parameter_type>;
};

template <typename... Containers>
struct buffers_to_parameter_integral_constant {
    using type = decltype(std::tuple_cat(
        std::tuple<typename parameter_types_to_integral_constant<Containers::parameter_type>::type>{}...));
};

template <ParameterType... ParameterTypes>
struct parameters_to_integral_constants {
    using type =
        decltype(std::tuple_cat(std::tuple<typename parameter_types_to_integral_constant<ParameterTypes>::type>{}...));
};

} // namespace kamping::internal
