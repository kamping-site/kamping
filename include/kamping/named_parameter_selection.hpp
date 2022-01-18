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
#include <tuple>

#include "kamping/parameter_type_definitions.hpp"

namespace kamping {
namespace internal {

/// @addtogroup kamping_utility
/// @{


/// @brief Returns the Index parameter if the parameter type of Arg matches the requested parameter type. If not, this
/// fails to compile.
///
/// This is the base case of the recursion.
///
/// @tparam ParameterType to be searched for.
/// @tparam Index index of current argument to evaluate.
/// @tparam Arg argument to evaluate.
/// @return the index
///
template <ParameterType parameter_type, size_t Index, typename Arg>
constexpr size_t find_pos() {
    // when we do not find the parameter type here, it is not given
    // a we fail to compile with a useful message
    static_assert(
        std::remove_reference_t<Arg>::parameter_type == parameter_type, "Could not find the requested parameter type.");
    return Index;
}

/// @brief Returns position of first argument in Args with Trait trait.
///
/// @tparam ParameterType to be searched for.
/// @tparam Index index of current argument to evaluate.
/// @tparam Arg argument to evaluate.
/// @tparam Arg2 the next argument.
/// @tparam Args all remaining arguments.
/// @return position of first argument with matched trait.
///
template <ParameterType parameter_type, size_t Index, typename Arg, typename Arg2, typename... Args>
constexpr size_t find_pos() {
    if constexpr (std::remove_reference_t<Arg>::parameter_type == parameter_type)
        return Index;
    else
        // we need to unpack the next two arguments, so we can unambiguously check for the case
        // of a single remaining argument
        return find_pos<parameter_type, Index + 1, Arg2, Args...>();
}
/// @brief Returns parameter with requested ParameterType.
///
/// @tparam ParameterType with which an argument should be found.
/// @tparam Args All arguments to be searched for an argument with ParameterType parameter_type.
/// @returns the first parameter whose type has the requested ParameterType.
///
template <ParameterType parameter_type, typename... Args>
decltype(auto) select_parameter_type(Args&&... args) {
    constexpr size_t selected_index = find_pos<parameter_type, 0, Args...>();
    using SelectedType              = typename std::tuple_element<selected_index, std::tuple<Args...>>::type;
    // TODO is this ok or too restricting?
    static_assert(
        std::is_lvalue_reference<SelectedType>::value,
        "Function does only accept lvalues, as it would produce dangling reference if called with temporaries");
    return std::forward<SelectedType>(std::get<selected_index>(std::forward_as_tuple(args...)));
}

///@}
} // namespace internal
} // namespace kamping
