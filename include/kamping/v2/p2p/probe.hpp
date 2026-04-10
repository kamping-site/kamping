#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/native_handle.hpp"

namespace kamping::core {

template <
    bridge::mpi_rank                                  Source = int,
    bridge::mpi_tag                                   Tag    = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void probe(
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) {
    int err = MPI_Probe(
        bridge::to_rank(source),
        bridge::to_tag(tag),
        bridge::native_handle(comm),
        bridge::native_handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core
