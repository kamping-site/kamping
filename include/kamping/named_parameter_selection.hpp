// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.:

/// @file
/// @brief Template magic to implement named parameters in cpp

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>

#include "kamping/parameter_type_definitions.hpp"

namespace kamping::internal {
/// @addtogroup kamping_utility
/// @{

/// @brief Returns the Index parameter if the parameter type of Arg matches the requested parameter type. If not, this
/// fails to compile.
///
/// This is the base case of the recursion.
///
/// @tparam parameter_type to be searched for.
/// @tparam Index index of current argument to evaluate.
/// @tparam Arg argument to evaluate.
/// @return the index
template <ParameterType parameter_type, size_t Index, typename Arg>
constexpr size_t find_pos() {
    constexpr bool found_arg = std::remove_reference_t<Arg>::parameter_type == parameter_type;
    // when we do not find the parameter type here, it is not given
    // a we fail to compile with a useful message
    return found_arg ? Index : std::numeric_limits<std::size_t>::max();
}

/// @brief Returns position of first argument in Args with Trait trait.
///
/// @tparam parameter_type to be searched for.
/// @tparam Index index of current argument to evaluate.
/// @tparam Arg argument to evaluate.
/// @tparam Arg2 the next argument.
/// @tparam Args all remaining arguments.
/// @return position of first argument with matched trait.
template <ParameterType parameter_type, size_t Index, typename Arg, typename Arg2, typename... Args>
constexpr size_t find_pos() {
    if constexpr (std::remove_reference_t<Arg>::parameter_type == parameter_type)
        return Index;
    else
        // we need to unpack the next two arguments, so we can unambiguously check for the case
        // of a single remaining argument
        return find_pos<parameter_type, Index + 1, Arg2, Args...>();
}

/// @brief Returns parameter with requested parameter type.
///
/// @tparam parameter_type with which a parameter should be found.
/// @tparam Args all parameter types to be searched for type `parameter_type`.
/// @param args all parameters from which a parameter with the correct type is selected.
/// @returns the first parameter whose type has the requested parameter type.
template <ParameterType parameter_type, typename... Args>
auto& select_parameter_type(Args&... args) {
    constexpr size_t selected_index = find_pos<parameter_type, 0, Args...>();
    static_assert(selected_index < sizeof...(args), "Could not find the requested parameter type.");
    return std::get<selected_index>(std::forward_as_tuple(args...));
}

/// @brief Checks if parameter with requested parameter type exists.
///
/// @tparam parameter_type with which a parameter should be found.
/// @tparam Args all parameter types to be searched.
/// @param args all parameter values.
/// @return whether `Args` contains a parameter of type `parameter_type`.
template <ParameterType parameter_type, typename... Args>
bool has_parameter_type(const Args&...) {
    return find_pos<parameter_type, 0, Args...>() < sizeof...(Args);
}

/// @brief Checks if parameter with requested parameter type exists at compile time.
///
/// @tparam parameter_type with which a parameter should be found.
/// @tparam Args all parameter types to be searched.
/// @return whether `Args` contains a parameter of type `parameter_type`.
template <ParameterType parameter_type, typename... Args>
constexpr bool has_parameter_type() {
    return find_pos<parameter_type, 0, Args...>() < sizeof...(Args);
}

/// @brief Checks if parameter with requested parameter type exists, if not constructs a default value.
///
/// @tparam parameter_type with which a parameter should be found.
/// @tparam Args all parameter types to be searched.
/// @tparam DefaultParameterType the type of the default parameter to be constructed.
/// @tparam DefaultArguments the types of parameters passed to the constructor \c DefaultParameterType.
/// @param default_arguments tuple of the arguments passed to the constructor of \c DefaultParameterType.
/// @param args all parameters from which a parameter with the correct type is selected.
/// @returns the first parameter whose type has the requested parameter type or the constructed default parameter if
/// none is found.
template <ParameterType parameter_type, typename DefaultParameterType, typename... DefaultArguments, typename... Args>
decltype(auto) select_parameter_type_or_default(std::tuple<DefaultArguments...> default_arguments, Args&... args) {
    if constexpr (has_parameter_type<parameter_type, Args...>()) {
        constexpr size_t selected_index = find_pos<parameter_type, 0, Args...>();
        return std::get<selected_index>(std::forward_as_tuple(args...));
    } else {
        return std::make_from_tuple<DefaultParameterType>(std::move(default_arguments));
    }
}
/// @}
} // namespace kamping::internal
