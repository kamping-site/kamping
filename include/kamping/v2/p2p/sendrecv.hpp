#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/result.hpp"

namespace kamping::core {

template <
    ranges::send_buffer                               SBuf,
    ranges::recv_buffer                               RBuf,
    bridge::mpi_rank                                  Dest    = int,
    bridge::mpi_rank                                  Source  = int,
    bridge::mpi_tag                                   SendTag = int,
    bridge::mpi_tag                                   RecvTag = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>       Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status  = MPI_Status*>
void sendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Status&&    status
) {
    int err = MPI_Sendrecv(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(send_tag),
        kamping::ranges::data(rbuf),
        static_cast<int>(kamping::ranges::size(rbuf)),
        kamping::ranges::type(rbuf),
        kamping::bridge::to_rank(source),
        kamping::bridge::to_tag(recv_tag),
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
    ranges::send_buffer                               SBuf,
    ranges::recv_buffer                               RBuf,
    bridge::mpi_rank                                  Dest    = int,
    bridge::mpi_rank                                  Source  = int,
    bridge::mpi_tag                                   SendTag = int,
    bridge::mpi_tag                                   RecvTag = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>       Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status  = MPI_Status*>
auto sendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source   = MPI_ANY_SOURCE,
    RecvTag     recv_tag = MPI_ANY_TAG,
    Comm const& comm     = MPI_COMM_WORLD,
    Status&&    status   = MPI_STATUS_IGNORE
) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(
        comm_op::sendrecv{},
        res.send,
        res.recv,
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(send_tag),
        kamping::bridge::to_rank(source),
        kamping::bridge::to_tag(recv_tag),
        kamping::bridge::native_handle(comm)
    );
    core::sendrecv(
        res.send,
        std::move(dest),
        std::move(send_tag),
        res.recv,
        std::move(source),
        std::move(recv_tag),
        comm,
        std::forward<Status>(status)
    );
    return res;
}

template <
    ranges::send_buffer                               SBuf,
    ranges::recv_buffer                               RBuf,
    bridge::mpi_rank                                  Dest   = int,
    bridge::mpi_rank                                  Source = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
auto sendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) -> result<SBuf, RBuf> {
    return sendrecv(
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        std::move(source),
        DEFAULT_SEND_TAG,
        comm,
        std::forward<Status>(status)
    );
}

} // namespace kamping::v2
