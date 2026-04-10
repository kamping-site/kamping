#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/native_handle.hpp"

namespace kamping::core {

template <
    bridge::mpi_rank                                   Source = int,
    bridge::mpi_tag                                    Tag    = int,
    bridge::convertible_to_mpi_handle_ptr<MPI_Message> Message,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm   = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Status>  Status = MPI_Status*>
void mprobe(Source source, Tag tag, Comm const& comm, Message&& message, Status&& status = MPI_STATUS_IGNORE) {
    int err = MPI_Mprobe(
        bridge::to_rank(source),
        bridge::to_tag(tag),
        bridge::native_handle(comm),
        bridge::native_handle_ptr(message),
        bridge::native_handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core
