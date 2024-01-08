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

#pragma once

#include <mpi.h>

#include "kamping/parameter_objects.hpp"
#include "kamping/status.hpp"

namespace kamping {
/// @addtogroup kamping_mpi_utility
/// @{

/// @brief Outputs the return status of the operation to the native \c MPI_Status passed by reference.
/// @param mpi_status The status.
template <typename StatusObject>
inline auto status_out(StatusObject&& status) {
    using status_type = std::remove_cv_t<std::remove_reference_t<StatusObject>>;
    static_assert(std::is_same_v<status_type, MPI_Status> || std::is_same_v<status_type, Status>);
    return internal::make_data_buffer<
        internal::ParameterType::status,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        no_resize>(std::forward<StatusObject>(status));
}

// /// @brief Outputs the return status of the operation to the provided \ref kamping::Status passed by reference.
// /// @brief Use the  provided \ref kamping::Status as status parameter.
// /// @param mpi_status The status.
// inline auto status_out(Status& mpi_status) {
//     return internal::StatusParam<internal::StatusParamType::ref>(mpi_status);
// }

/// @brief Constructs a status object internally, which may then be retrieved from \c kamping::MPIResult returned by the
/// operation.
inline auto status_out() {
    return internal::make_data_buffer<
        internal::ParameterType::status,
        internal::BufferModifiability::modifiable,
        internal::BufferType::out_buffer,
        no_resize>(alloc_new<Status>);
}

/// @brief pass \c MPI_STATUS_IGNORE to the underlying MPI call.
inline auto status(internal::ignore_t<void>) {
    return internal::EmptyDataBuffer<Status, internal::ParameterType::status, internal::BufferType::ignore>{};
}

/// @}
} // namespace kamping
