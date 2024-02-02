// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <cstddef>

#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/mpi_datatype.hpp"

namespace kamping {

/// @brief Wrapper for MPI_Status
class Status {
public:
    /// @brief Construct a status object. Note that all values are undefined until passed to a communication function.
    Status() : _status() {}
    /// @brief Construct a status object from a given MPI_Status.
    /// @param status The status.
    Status(MPI_Status status) : _status(std::move(status)) {}

    /// @return The source rank. May be undefined.
    [[nodiscard]] int source_signed() const {
        return _status.MPI_SOURCE;
    }

    /// @return The source rank. May be undefined.
    [[nodiscard]] size_t source() const {
        return asserting_cast<size_t>(source_signed());
    }

    /// @return The tag. May be undefined.
    [[nodiscard]] int tag() const {
        return _status.MPI_TAG;
    }

    /// @param data_type The datatype.
    /// @return The number of top-level elements received for the given type \c
    /// DataType.
    [[nodiscard]] int count_signed(MPI_Datatype data_type) const {
        int count;
        MPI_Get_count(&_status, data_type, &count);
        return count;
    }

    /// @tparam DataType The datatype.
    /// @return The number of top-level elements received for the given type \c
    /// DataType.
    template <typename DataType>
    [[nodiscard]] int count_signed() const {
        return this->count_signed(mpi_datatype<DataType>());
    }

    /// @param data_type The datatype.
    /// @return The number of top-level elements received for the given type \c
    /// DataType.
    [[nodiscard]] size_t count(MPI_Datatype data_type) const {
        return asserting_cast<size_t>(this->count_signed(data_type));
    }

    /// @tparam DataType The datatype.
    /// @return The number of top-level elements received for the given type \c
    /// DataType.
    template <typename DataType>
    [[nodiscard]] size_t count() const {
        return asserting_cast<size_t>(this->count_signed<DataType>());
    }

    /// @return A reference to the underlying native MPI_Status.
    [[nodiscard]] MPI_Status& native() {
        return _status;
    }

    /// @return A reference to the underlying native MPI_Status.
    [[nodiscard]] MPI_Status const& native() const {
        return _status;
    }

private:
    MPI_Status _status; ///< The wrapped status.
};
} // namespace kamping
