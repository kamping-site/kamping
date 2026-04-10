#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/iresult.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/p2p/imrecv.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"

namespace kamping::core {
template <
    ranges::recv_buffer                                RBuf,
    bridge::mpi_rank                                   Source  = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void irecv(RBuf&& rbuf, Source source, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Irecv(
        kamping::ranges::data(rbuf),
        static_cast<int>(kamping::ranges::size(rbuf)),
        kamping::ranges::type(rbuf),
        kamping::bridge::to_rank(source),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm),
        kamping::bridge::native_handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core

namespace kamping::v2 {
template <
    ranges::recv_buffer                                RBuf,
    bridge::mpi_rank                                   Source  = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto irecv(
    Request&&   request,
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD
) -> RBuf {
    constexpr bool infer_returns_mpi_message =
        requires(RBuf&& rbuf_, Source source_, Tag tag_, Comm const& comm_) {
        {
            infer(
                comm_op::recv{},
                rbuf_,
                kamping::bridge::to_rank(source_),
                kamping::bridge::to_tag(tag_),
                kamping::bridge::native_handle(comm_)
            )
        } -> std::same_as<MPI_Message>;
    };
    if constexpr (infer_returns_mpi_message) {
        MPI_Message msg = infer(
            comm_op::recv{},
            rbuf,
            kamping::bridge::to_rank(source),
            kamping::bridge::to_tag(tag),
            kamping::bridge::native_handle(comm)
        );
        core::imrecv(rbuf, &msg, std::forward<Request>(request));
        return std::forward<RBuf>(rbuf);
    } else {
        infer(
            comm_op::recv{},
            rbuf,
            kamping::bridge::to_rank(source),
            kamping::bridge::to_tag(tag),
            kamping::bridge::native_handle(comm)
        );
        core::irecv(rbuf, std::move(source), std::move(tag), comm, std::forward<Request>(request));
        return std::forward<RBuf>(rbuf);
    }
}
/// Issues a non-blocking receive and returns an iresult that can be waited on.
///
/// **Synchronous work on the caller's thread.** Before the underlying MPI_Irecv/
/// MPI_Imrecv is issued, this call invokes the `infer(comm_op::recv, ...)`
/// customization point. The default overload for buffers satisfying
/// `resizable_recv_buf` runs a **blocking** `MPI_Mprobe` so the incoming message
/// size can be determined and the buffer sized to fit. User-supplied `infer()`
/// overloads may also perform blocking work. In both cases, this call will not
/// return until that synchronous work has completed — it is "non-blocking" only
/// for the wire-level data transfer, not for any handshake that precedes it.
///
/// If you need an `irecv` that is guaranteed not to block on the caller's thread,
/// pass a buffer whose `infer()` overload is a no-op (for example, a fixed-size
/// recv buffer that the caller has already sized correctly).
template <
    ranges::recv_buffer                         RBuf,
    bridge::mpi_rank                            Source = int,
    bridge::mpi_tag                             Tag    = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm   = MPI_Comm>
auto irecv(RBuf&& rbuf, Source source = MPI_ANY_SOURCE, Tag tag = MPI_ANY_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> iresult<RBuf> {
    iresult<RBuf> res{std::forward<RBuf>(rbuf)};
    using view_t                           = ranges::all_t<RBuf>;
    constexpr bool infer_returns_mpi_message =
        requires(view_t& v, Source source_, Tag tag_, Comm const& comm_) {
            {
                infer(
                    comm_op::recv{},
                    v,
                    kamping::bridge::to_rank(source_),
                    kamping::bridge::to_tag(tag_),
                    kamping::bridge::native_handle(comm_)
                )
            } -> std::same_as<MPI_Message>;
        };
    if constexpr (infer_returns_mpi_message) {
        MPI_Message msg = infer(
            comm_op::recv{},
            res.view(),
            kamping::bridge::to_rank(source),
            kamping::bridge::to_tag(tag),
            kamping::bridge::native_handle(comm)
        );
        core::imrecv(res.view(), &msg, res.mpi_native_handle_ptr());
    } else {
        infer(
            comm_op::recv{},
            res.view(),
            kamping::bridge::to_rank(source),
            kamping::bridge::to_tag(tag),
            kamping::bridge::native_handle(comm)
        );
        core::irecv(res.view(), source, tag, comm, res.mpi_native_handle_ptr());
    }
    return res;
}
} // namespace kamping::v2
