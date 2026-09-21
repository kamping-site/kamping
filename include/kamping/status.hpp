// This file is part of KaMPIng.
//
// Copyright 2022-2024 The KaMPIng Authors
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

template <typename StatusType>
class StatusBase {
private:
    MPI_Status* status_ptr() {
        return static_cast<StatusType&>(*this).status_ptr();
    }

    MPI_Status const* status_ptr() const {
        return static_cast<StatusType const&>(*this).status_ptr();
    }

public:
    ///
    /// @return The source rank. May be undefined.
    [[nodiscard]] int source_signed() const {
        return status_ptr()->MPI_SOURCE;
    }

    /// @return The source rank. May be undefined.
    [[nodiscard]] size_t source() const {
        return asserting_cast<size_t>(source_signed());
    }

    /// @return The tag. May be undefined.
    [[nodiscard]] int tag() const {
        return status_ptr()->MPI_TAG;
    }

    /// @param data_type The datatype.
    /// @return The number of top-level elements received for the given type \c
    /// DataType.
    [[nodiscard]] int count_signed(MPI_Datatype data_type) const {
        int count;
        MPI_Get_count(status_ptr(), data_type, &count);
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
        return *status_ptr();
    }

    /// @return A reference to the underlying native MPI_Status.
    [[nodiscard]] MPI_Status const& native() const {
        return *status_ptr();
    }
};

/// @brief Wrapper for MPI_Status
class Status : public StatusBase<Status> {
    friend class StatusBase<Status>;

private:
    MPI_Status* status_ptr() {
        return &_status;
    }

    MPI_Status const* status_ptr() const {
        return &_status;
    }

public:
    /// @brief Construct a status object. Note that all values are undefined until passed to a communication function.
    Status() : _status() {}
    /// @brief Construct a status object from a given MPI_Status.
    /// @param status The status.
    Status(MPI_Status status) : _status(std::move(status)) {}

private:
    MPI_Status _status; ///< The wrapped status.
};

/// @brief Wrapper for MPI_Status
class StatusConstRef : public StatusBase<StatusConstRef> {
    friend class StatusBase<StatusConstRef>;

private:
    MPI_Status* status_ptr() = delete;

    MPI_Status const* status_ptr() const {
        return &_status;
    }

public:
    /// @brief Construct a status object from a given MPI_Status.
    /// @param status The status.
    StatusConstRef(MPI_Status const& status) : _status(status) {}

private:
    MPI_Status const& _status; ///< The wrapped status.
};

} // namespace kamping
