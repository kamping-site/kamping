// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
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
/// @brief Wrapper for MPI constants

#pragma once

#include <mpi.h>
namespace kamping {
///@brief Wraps the possible results when calling \c MPI_Comm_compare on two MPI communicators comm1 and comm2
enum class CommunicatorComparisonResult : int {
    identical = MPI_IDENT,     ///< Result if comm1 and comm2 are handles for same object.
    congruent = MPI_CONGRUENT, ///< Result if the underlying groups of comm1 and comm2 are identical but the
                               ///< communicators's context is different.
    similar = MPI_SIMILAR,     ///< Result if the underlying groups contain the same ranks but their order differs.
    unequal = MPI_UNEQUAL      ///< Result otherwise.
};
} // namespace kamping
