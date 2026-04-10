#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/iresult.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/result.hpp"

namespace kamping::core {

template <
    ranges::send_buffer                                SBuf,
    ranges::recv_buffer                                RBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_rank                                   Source  = int,
    bridge::mpi_tag                                    SendTag = int,
    bridge::mpi_tag                                    RecvTag = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void isendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source,
    RecvTag     recv_tag,
    Comm const& comm,
    Request&&   request
) {
    int err = MPI_Isendrecv(
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
        kamping::bridge::native_handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

} // namespace kamping::core

namespace kamping::v2 {

/// Low-level overload: caller supplies an external request and manages its lifetime.
/// Returns result<SBuf, RBuf> immediately; the operation may still be in flight.
template <
    ranges::send_buffer                                SBuf,
    ranges::recv_buffer                                RBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_rank                                   Source  = int,
    bridge::mpi_tag                                    SendTag = int,
    bridge::mpi_tag                                    RecvTag = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto isendrecv(
    Request&&   request,
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source   = MPI_ANY_SOURCE,
    RecvTag     recv_tag = MPI_ANY_TAG,
    Comm const& comm     = MPI_COMM_WORLD
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
    core::isendrecv(
        res.send,
        std::move(dest),
        std::move(send_tag),
        res.recv,
        std::move(source),
        std::move(recv_tag),
        comm,
        std::forward<Request>(request)
    );
    return res;
}

/// Low-level overload without explicit tags (uses DEFAULT_SEND_TAG).
template <
    ranges::send_buffer                                SBuf,
    ranges::recv_buffer                                RBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_rank                                   Source  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto isendrecv(
    Request&&   request,
    SBuf&&      sbuf,
    Dest        dest,
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Comm const& comm   = MPI_COMM_WORLD
) -> result<SBuf, RBuf> {
    return isendrecv(
        std::forward<Request>(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        std::move(source),
        DEFAULT_SEND_TAG,
        comm
    );
}

/// High-level overload: creates and owns the MPI_Request internally.
/// Returns iresult<SBuf, RBuf>; call wait() to block and retrieve the result.
///
/// **Synchronous work on the caller's thread.** Before the underlying
/// MPI_Isendrecv is issued, this call invokes the `infer(comm_op::sendrecv, ...)`
/// customization point. The default overload for buffers satisfying
/// `resizable_recv_buf` runs a **blocking** `MPI_Sendrecv` of a single count so
/// the incoming message size can be determined and the recv buffer sized to fit.
/// User-supplied `infer()` overloads may also perform blocking work. In both
/// cases, this call will not return until that synchronous work has completed —
/// it is "non-blocking" only for the data-phase MPI_Isendrecv, not for any
/// handshake that precedes it.
///
/// If you need an `isendrecv` that is guaranteed not to block on the caller's
/// thread, pass a recv buffer whose `infer()` overload is a no-op (for example,
/// a fixed-size recv buffer that the caller has already sized correctly).
template <
    ranges::send_buffer                         SBuf,
    ranges::recv_buffer                         RBuf,
    bridge::mpi_rank                            Dest    = int,
    bridge::mpi_rank                            Source  = int,
    bridge::mpi_tag                             SendTag = int,
    bridge::mpi_tag                             RecvTag = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm    = MPI_Comm>
auto isendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    SendTag     send_tag,
    RBuf&&      rbuf,
    Source      source   = MPI_ANY_SOURCE,
    RecvTag     recv_tag = MPI_ANY_TAG,
    Comm const& comm     = MPI_COMM_WORLD
) -> iresult<SBuf, RBuf> {
    iresult<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(
        comm_op::sendrecv{},
        res.send(),
        res.recv(),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(send_tag),
        kamping::bridge::to_rank(source),
        kamping::bridge::to_tag(recv_tag),
        kamping::bridge::native_handle(comm)
    );
    core::isendrecv(
        res.send(),
        dest,
        send_tag,
        res.recv(),
        source,
        recv_tag,
        comm,
        res.mpi_native_handle_ptr()
    );
    return res;
}

/// High-level overload without explicit tags (uses DEFAULT_SEND_TAG).
template <
    ranges::send_buffer                         SBuf,
    ranges::recv_buffer                         RBuf,
    bridge::mpi_rank                            Dest   = int,
    bridge::mpi_rank                            Source = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm   = MPI_Comm>
auto isendrecv(
    SBuf&&      sbuf,
    Dest        dest,
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Comm const& comm   = MPI_COMM_WORLD
) -> iresult<SBuf, RBuf> {
    return isendrecv(
        std::forward<SBuf>(sbuf),
        std::move(dest),
        DEFAULT_SEND_TAG,
        std::forward<RBuf>(rbuf),
        std::move(source),
        DEFAULT_SEND_TAG,
        comm
    );
}

} // namespace kamping::v2
