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

#include <kamping/mpi_datatype.hpp>

namespace kamping {
/// @brief Specialization of the `mpi_type_traits` type trait for `std::tuple`, which represents the tuple as a sequence
/// of bytes.
/// @warning This is UB, because byte serialization is only well-defined for trivially copyable types. For a safe
/// version see \ref kamping/types/tuple.hpp.
template <typename... Ts>
struct mpi_type_traits<std::tuple<Ts...>, std::enable_if_t<(has_static_type_v<Ts> && ...)>>
    : byte_serialized<std::tuple<Ts...>> {};
} // namespace kamping
