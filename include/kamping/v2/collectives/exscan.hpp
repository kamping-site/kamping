#pragma once

#include <functional>
#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/result.hpp"
#include "mpi/collectives/exscan.hpp"
#include "mpi/handle.hpp"
#include "mpi/ops.hpp"

namespace kamping::v2 {

/// Exclusive prefix scan (blocking, two-buffer form).
///
/// Computes an exclusive prefix reduction and places the result in the receive buffer.
/// All ranks are symmetric (no root).
///
/// @note The receive buffer content on rank 0 is left undefined by MPI. v2 does not
///       fill rank 0's buffer — this is application logic. Document and handle at
///       the call site if needed.
///
/// @tparam SBuf Send buffer type satisfying send_buffer
/// @tparam RBuf Receive buffer type satisfying recv_buffer; may be a resizable view (views::resize).
/// @tparam Op   Operation type satisfying valid_op<Op, SBuf, RBuf> (default: std::plus<>)
/// @tparam Comm Communicator type (default: MPI_Comm)
///
/// @param sbuf Send buffer; pass kamping::v2::inplace to use rbuf for both send and recv.
/// @param rbuf Receive buffer. Content on rank 0 is undefined after the call.
/// @param op   Reduction operation (MPI_Op or builtin operation, default: std::plus<>).
/// @param comm MPI communicator (default: MPI_COMM_WORLD).
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::valid_op<SBuf, RBuf>                Op   = std::plus<>,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto exscan(SBuf&& sbuf, RBuf&& rbuf, Op const& op = std::plus<>{}, Comm const& comm = MPI_COMM_WORLD)
    -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(
        comm_op::exscan{},
        res.send,
        res.recv,
        mpi::experimental::as_mpi_op(op, res.send, res.recv),
        mpi::experimental::handle(comm)
    );
    mpi::experimental::exscan(res.send, res.recv, op, comm);
    return res;
}

} // namespace kamping::v2
