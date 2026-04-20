#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"
#include "mpi/ops.hpp"

namespace mpi::experimental {

/// Inclusive prefix scan (blocking).
///
/// Computes a prefix reduction on the send buffer of each process and places
/// the result in the receive buffer. All ranks are symmetric (no root).
///
/// @tparam SBuf Send buffer type satisfying send_buffer
/// @tparam RBuf Receive buffer type satisfying recv_buffer
/// @tparam Op   Operation type satisfying valid_op<Op, SBuf, RBuf>
/// @tparam Comm Communicator type convertible to MPI_Comm
///
/// @param sbuf Send buffer; pass MPI_IN_PLACE to use rbuf for both send and recv.
/// @param rbuf Receive buffer.
/// @param op   Reduction operation.
/// @param comm MPI communicator (default: MPI_COMM_WORLD).
template <
    send_buffer                                            SBuf,
    recv_buffer                                            RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void scan(SBuf&& sbuf, RBuf&& rbuf, Op const& op, Comm const& comm = MPI_COMM_WORLD) {
    auto sbuf_ptr = ptr(sbuf);
    if (sbuf_ptr == MPI_IN_PLACE) {
        int err = MPI_Scan(
            sbuf_ptr,
            ptr(rbuf),
            static_cast<int>(count(rbuf)),
            type(rbuf),
            as_mpi_op(op, sbuf, rbuf),
            handle(comm)
        );
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    } else {
        using scount_t = decltype(count(sbuf));
        KAMPING_ASSERT(
            count(sbuf) == static_cast<scount_t>(count(rbuf)),
            "send and receive buffer must have the same count"
        );
        KAMPING_ASSERT(type(sbuf) == type(rbuf), "send and receive buffer must have the same type");
        int err = MPI_Scan(
            sbuf_ptr,
            ptr(rbuf),
            static_cast<int>(count(sbuf)),
            type(sbuf),
            as_mpi_op(op, sbuf, rbuf),
            handle(comm)
        );
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
    }
}

} // namespace mpi::experimental
