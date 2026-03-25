// This file is part of KaMPIng.
//
// Copyright 2021-2025 The KaMPIng Authors
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
/// @brief Contiguous and byte-serialized MPI type helpers.

#pragma once
#include <cstddef>
#include <type_traits>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "kamping/types/builtin_types.hpp"
#include "kamping/types/mpi_type_traits.hpp"

namespace kamping {

/// @brief Constructs a contiguous MPI type of \p N elements of type \p T using `MPI_Type_contiguous`.
template <typename T, size_t N>
struct contiguous_type {
    static constexpr TypeCategory category          = TypeCategory::contiguous; ///< The category of the type.
    static constexpr bool         has_to_be_committed = category_has_to_be_committed(category); ///< Whether commit is needed.
    /// @brief The MPI_Datatype corresponding to the type.
    static MPI_Datatype data_type();
};

/// @brief Constructs a type serialized as a sequence of `sizeof(T)` bytes using `MPI_BYTE`.
/// Note that you must ensure that this conversion is valid.
template <typename T>
struct byte_serialized : contiguous_type<std::byte, sizeof(T)> {};

template <typename T, size_t N>
MPI_Datatype contiguous_type<T, N>::data_type() {
    MPI_Datatype type;
    MPI_Datatype base_type;
    if constexpr (std::is_same_v<T, std::byte>) {
        base_type = MPI_BYTE;
    } else {
        static_assert(
            has_static_type_v<T>,
            "\n --> Type not supported directly by KaMPIng. Please provide a specialization for mpi_type_traits."
        );
        base_type = mpi_type_traits<T>::data_type();
    }
    int const count = static_cast<int>(N);
    int const err   = MPI_Type_contiguous(count, base_type, &type);
    KAMPING_ASSERT(err == MPI_SUCCESS, "MPI_Type_contiguous failed");
    return type;
}

} // namespace kamping
