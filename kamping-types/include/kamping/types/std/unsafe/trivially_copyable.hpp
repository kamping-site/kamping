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

#include "kamping/types/contiguous_type.hpp"
#include "kamping/types/detail/type_helpers.hpp"
#include "kamping/types/mpi_type_traits.hpp"

namespace kamping::types {

/// @addtogroup kamping_types
/// @{

/// @brief Opt-in catch-all specialization of \ref kamping::types::mpi_type_traits for any
///        trivially copyable type not already covered by the built-in dispatcher.
///
/// Represents the object as a flat sequence of `sizeof(T)` bytes using `MPI_BYTE`.
///
/// `std::pair` and `std::tuple` are excluded so this header composes safely with
/// \ref kamping/types/std/utility.hpp, \ref kamping/types/std/tuple.hpp, and their unsafe
/// counterparts — include whichever combination suits your use case.
///
/// @warning Padding bytes between fields are silently included in the byte representation.
///          This is correct for tightly-packed structs, but incorrect for structs with
///          compiler-inserted padding holes. Users opt in to this trade-off knowingly by
///          including this header.
template <typename T>
struct mpi_type_traits<
    T,
    std::enable_if_t<
        std::is_trivially_copyable<T>::value && !has_auto_dispatched_type_v<T>
        && !kamping::internal::is_std_pair<T>::value && !kamping::internal::is_std_tuple<T>::value>>
    : byte_serialized<T> {};

/// @}

} // namespace kamping::types
