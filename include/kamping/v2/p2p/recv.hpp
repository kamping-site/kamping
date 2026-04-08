#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"

namespace kamping::core {
template <
    ranges::recv_buffer                               RBuf,
    bridge::mpi_rank                                  Source = int,
    bridge::mpi_tag                                   Tag    = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void recv(
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) {
    int err = MPI_Recv(
        kamping::ranges::data(rbuf),
        static_cast<int>(kamping::ranges::size(rbuf)),
        kamping::ranges::type(rbuf),
        kamping::bridge::to_rank(source),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm),
        kamping::bridge::native_handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core

namespace kamping::v2 {

template <
    ranges::recv_buffer                               RBuf,
    bridge::mpi_rank                                  Source = int,
    bridge::mpi_tag                                   Tag    = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto recv(
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) -> ranges::buf_result_t<RBuf> {
    infer(
        comm_op::recv{},
        rbuf,
        kamping::bridge::to_rank(source),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm)
    );
    if constexpr (!std::is_reference_v<RBuf> && !ranges::borrowed_buffer<RBuf>) {
        auto buf = std::move(rbuf);
        core::recv(buf, std::move(source), std::move(tag), comm, status);
        return buf;
    } else {
        core::recv(rbuf, std::move(source), std::move(tag), comm, status);
        return std::forward<RBuf>(rbuf);
    }
}
template <
    ranges::recv_buffer                               RBuf,
    bridge::convertible_to_mpi_handle<MPI_Comm>       Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto recv(RBuf&& rbuf, Comm const& comm, Status&& status = MPI_STATUS_IGNORE) -> ranges::buf_result_t<RBuf> {
    return recv(std::forward<RBuf>(rbuf), MPI_ANY_SOURCE, MPI_ANY_TAG, comm, std::forward<Status>(status));
}
} // namespace kamping::v2
