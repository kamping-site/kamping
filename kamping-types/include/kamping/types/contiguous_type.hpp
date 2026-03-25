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
/// @brief `MPI_Type_contiguous` implementation for \ref kamping::types::contiguous_type and \ref
/// kamping::types::byte_serialized.

#pragma once
#include <cstddef>
#include <type_traits>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
// contiguous_type and byte_serialized structs are declared in the fwd header;
// include it here so consumers only need this single header.
#include "kamping/types/detail/contiguous_type_fwd.hpp"
#include "kamping/types/mpi_type_traits.hpp"

namespace kamping::types {

/// @addtogroup kamping_types
/// @{

template <typename T, size_t N, typename Lookup>
MPI_Datatype contiguous_type<T, N, Lookup>::data_type() {
    MPI_Datatype type;
    MPI_Datatype base_type;
    if constexpr (std::is_same_v<T, std::byte>) {
        base_type = MPI_BYTE;
    } else {
        static_assert(
            Lookup::template has_type_v<T>,
            "\n --> Type not supported by the current Lookup policy. "
            "Please specialize mpi_type_traits for this type or provide a custom Lookup."
        );
        base_type = Lookup::template get<T>();
    }
    int const count = static_cast<int>(N);
    int const err   = MPI_Type_contiguous(count, base_type, &type);
    KAMPING_ASSERT(err == MPI_SUCCESS, "MPI_Type_contiguous failed");
    return type;
}

/// @}

} // namespace kamping::types
