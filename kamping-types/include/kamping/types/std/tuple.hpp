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

#include <tuple>
#include <type_traits>

#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/types/struct_type.hpp"

namespace kamping::types {

/// @brief Specialization of \ref kamping::types::mpi_type_traits for `std::tuple`, representing
///        the tuple as an MPI struct type derived from its element types.
///
/// All element types must already have a static MPI type (i.e. \ref kamping::types::has_static_type_v
/// must be `true` for every element type). The resulting MPI datatype correctly reflects the memory
/// layout, including any padding inserted by the compiler.
///
/// @note This header and \ref kamping/types/std/unsafe/tuple.hpp provide conflicting
///       specializations — include only one.
///
/// @note For maximum throughput at the cost of correctness guarantees, use the byte-serialized
///       variant in \ref kamping/types/std/unsafe/tuple.hpp instead.
template <typename... Ts>
struct mpi_type_traits<std::tuple<Ts...>, std::enable_if_t<(has_static_type_v<Ts> && ...)>>
    : struct_type<std::tuple<Ts...>> {};

} // namespace kamping::types
