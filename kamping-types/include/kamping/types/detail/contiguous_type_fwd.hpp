// This file is part of KaMPIng.
//
// Copyright 2021-2026 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief Forward declarations for contiguous_type and byte_serialized to break include cycles.

#pragma once
#include <cstddef>

#include <mpi.h>

#include "kamping/types/builtin_types.hpp"

namespace kamping::types {

/// @addtogroup kamping_types
/// @{

/// @brief Forward declaration of the default lookup policy for \ref contiguous_type and \ref struct_type.
/// @see type_dispatcher_lookup
struct type_dispatcher_lookup;

/// @brief Constructs a contiguous MPI type of \p N elements of type \p T using `MPI_Type_contiguous`.
/// @tparam T The element type.
/// @tparam N The number of elements.
/// @tparam Lookup The lookup policy used to resolve the MPI_Datatype for \p T.
///   Defaults to \ref type_dispatcher_lookup, which uses \ref kamping::types::mpi_type_traits.
template <typename T, size_t N, typename Lookup = type_dispatcher_lookup>
struct contiguous_type {
    static constexpr TypeCategory category = TypeCategory::contiguous; ///< The type's \ref TypeCategory.
    static constexpr bool         has_to_be_committed =
        category_has_to_be_committed(category); ///< Whether the type must be committed before use.
    static MPI_Datatype data_type(); ///< Returns the MPI_Datatype for a contiguous block of \p N elements of type \p T.
};

/// @brief Constructs a type serialized as a sequence of `sizeof(T)` bytes using `MPI_BYTE`.
template <typename T>
struct byte_serialized : contiguous_type<std::byte, sizeof(T)> {};

/// @}

} // namespace kamping::types
