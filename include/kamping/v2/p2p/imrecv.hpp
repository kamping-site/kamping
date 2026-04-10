#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/ranges/concepts.hpp"
namespace kamping::core {
template <
    ranges::recv_buffer                                RBuf,
    bridge::convertible_to_mpi_handle_ptr<MPI_Message> Message,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request>
void imrecv(RBuf&& rbuf, Message&& message, Request&& request) {
    int err = MPI_Imrecv(
        kamping::ranges::data(rbuf),
        static_cast<int>(kamping::ranges::size(rbuf)),
        kamping::ranges::type(rbuf),
        bridge::native_handle_ptr(message),
        bridge::native_handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core
