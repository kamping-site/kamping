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
//
#pragma once

#include <type_traits>
#include <utility>

#include "kamping/types/contiguous_type.hpp"
#include "kamping/types/mpi_type_traits.hpp"

namespace kamping::types {

/// @brief Specialization of \ref kamping::types::mpi_type_traits for `std::pair`, representing
///        the pair as a flat sequence of `sizeof(std::pair<First, Second>)` bytes using `MPI_BYTE`.
///
/// Both field types must already have a static MPI type (i.e. \ref kamping::types::has_static_type_v
/// must be `true` for both `First` and `Second`).
///
/// @warning This serialization ignores any padding bytes the compiler may insert between or after
///          fields. Sending a padded pair is only correct when both sides share the same ABI. For
///          a layout-safe alternative use \ref kamping/types/std/utility.hpp.
///
/// @note This header and \ref kamping/types/std/utility.hpp provide conflicting
///       specializations — include only one.
template <typename First, typename Second>
struct mpi_type_traits<
    std::pair<First, Second>,
    std::enable_if_t<has_static_type_v<First> && has_static_type_v<Second>>>
    : byte_serialized<std::pair<First, Second>> {};

} // namespace kamping::types
