#pragma once

#include <functional>
#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/scan.hpp"
#include "mpi/handle.hpp"
#include "mpi/ops.hpp"

namespace kamping::v2 {

/// Inclusive prefix scan (blocking, two-buffer form).
///
/// Computes a prefix reduction on the send buffer of each process and places
/// the result in the receive buffer. All ranks are symmetric (no root).
///
/// @tparam SBuf Send buffer type satisfying send_buffer
/// @tparam RBuf Receive buffer type satisfying recv_buffer; may be a resizable view (views::resize).
/// @tparam Op   Operation type satisfying valid_op<Op, SBuf, RBuf> (default: std::plus<>)
/// @tparam Comm Communicator type (default: MPI_Comm)
///
/// @param sbuf Send buffer; pass kamping::v2::inplace to use rbuf for both send and recv.
/// @param rbuf Receive buffer.
/// @param op   Reduction operation (MPI_Op or builtin operation, default: std::plus<>).
/// @param comm MPI communicator (default: MPI_COMM_WORLD).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op   = std::plus<>,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto scan(SBuf&& sbuf, RBuf&& rbuf, Op const& op = std::plus<>{}, Comm const& comm = MPI_COMM_WORLD)
    -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(
        comm_op::scan{},
        res.send,
        res.recv,
        mpi::experimental::as_mpi_op(op, res.send, res.recv),
        mpi::experimental::handle(comm)
    );
    mpi::experimental::scan(res.send, res.recv, op, comm);
    return res;
}

} // namespace kamping::v2
