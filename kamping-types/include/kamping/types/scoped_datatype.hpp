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
/// @brief RAII wrapper that commits an \c MPI_Datatype on construction and frees it on destruction.

#pragma once
#include <utility>

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"

namespace kamping::types {

/// @addtogroup kamping_types
/// @{

/// @brief RAII wrapper that commits an \c MPI_Datatype on construction and frees it on destruction.
///
/// Calls \c MPI_Type_commit / \c MPI_Type_free directly and does not depend on the KaMPIng
/// environment. Useful for managing custom derived datatypes with automatic lifetime.
class ScopedDatatype {
    MPI_Datatype _type; ///< The MPI_Datatype.
public:
    /// @brief Construct a new scoped MPI_Datatype and commit it.
    /// If no type is provided, defaults to `MPI_DATATYPE_NULL` and does not commit or free anything.
    ScopedDatatype(MPI_Datatype type = MPI_DATATYPE_NULL) : _type(type) {
        // FIXME: ensure that we don't commit/free named types
        if (type != MPI_DATATYPE_NULL) {
            int const err = MPI_Type_commit(&_type);
            KAMPING_ASSERT(err == MPI_SUCCESS, "MPI_Type_commit failed");
        }
    }
    /// @brief Deleted copy constructor.
    ScopedDatatype(ScopedDatatype const&) = delete;
    /// @brief Deleted copy assignment.
    ScopedDatatype& operator=(ScopedDatatype const&) = delete;

    /// @brief Move constructor.
    ScopedDatatype(ScopedDatatype&& other) noexcept : _type(other._type) {
        other._type = MPI_DATATYPE_NULL;
    }
    /// @brief Move assignment.
    ScopedDatatype& operator=(ScopedDatatype&& other) noexcept {
        std::swap(_type, other._type);
        return *this;
    }
    /// @brief Get the MPI_Datatype.
    MPI_Datatype const& data_type() const {
        return _type;
    }
    /// @brief Free the MPI_Datatype.
    ~ScopedDatatype() {
        if (_type != MPI_DATATYPE_NULL) {
            int const err = MPI_Type_free(&_type);
            KAMPING_ASSERT(err == MPI_SUCCESS, "MPI_Type_free failed");
        }
    }
};

/// @}

} // namespace kamping::types
