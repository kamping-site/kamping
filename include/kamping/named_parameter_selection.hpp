// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.:


/// @file
/// @brief Template magic to implement named parameters in cpp

#pragma once

#include <cstddef>
#include <tuple>

#include "parameter_type_definitions.hpp"

namespace kamping {
namespace internal {

/// @addtogroup kamping_utility
/// @{


/// @brief returns position of first argument in Args with Trait trait and returns it
///
/// @tparam Trait trait with which an argument should be found
/// @tparam I index of current argument to evaluate
/// @tparam A argument to evaluate
/// @tparam Args all remaining arguments
/// @return position of first argument whit matched trait
///
template <ParameterType ptype, size_t I, typename Arg, typename... Args>
constexpr size_t find_pos() {
    if constexpr (Arg::ptype == ptype)
        return I;
    else
        return find_pos<ptype, I + 1, Args...>();
}
/// @brief returns parameter which the desired trait
///
/// @tparam Trait trait with which an argument should be found
/// @tparam Args all arguments to be searched
/// returns the first parameter whose type has the appropriate par_type
///
template <ParameterType ptype, typename... Args>
decltype(auto) select_ptype(Args&&... args) {
    return std::move(std::get<find_pos<ptype, 0, Args...>()>(std::forward_as_tuple(args...)));
}

///@}
} // namespace internal
} // namespace kamping
