#pragma once

#include <mpi.h>

#include "kamping/parameter_objects.hpp"
#include "kamping/status.hpp"

namespace kamping {
/// @brief Use the provided native \c MPI_Status as status parameter.
/// @param mpi_status The status.
inline auto status(MPI_Status& mpi_status) {
    return internal::StatusParam<internal::StatusParamType::native_ref>{mpi_status};
}

/// @brief Use the  provided \ref kamping::Status as status parameter.
/// @param mpi_status The status.
inline auto status(Status& mpi_status) {
    return internal::StatusParam<internal::StatusParamType::ref>(mpi_status);
}

/// @brief Construct a status object internally, which may then be retrieved from \c kamping::MPIResult returned by the
/// operation.
inline auto status_out() {
    return internal::StatusParam<internal::StatusParamType::owning>{};
}

/// @brief pass \c MPI_STATUS_IGNORE to the underlying MPI call.
inline auto status(internal::ignore_t<void>) {
    return internal::StatusParam<internal::StatusParamType::ignore>{};
}

} // namespace kamping
