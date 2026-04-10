#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/iresult.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/p2p/send_mode.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"

namespace kamping::core {
template <
    ranges::send_buffer                                SBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void isend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Isend(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm),
        kamping::bridge::native_handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    ranges::send_buffer                                SBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void ibsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Ibsend(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm),
        kamping::bridge::native_handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    ranges::send_buffer                                SBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void issend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Issend(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm),
        kamping::bridge::native_handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    ranges::send_buffer                                SBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void irsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Irsend(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
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

/// Low-level overload: caller supplies an external MPI_Request* and manages its lifetime.
/// Returns the buffer (ownership semantics match blocking send).
template <
    is_send_mode                                       SendMode,
    ranges::send_buffer                                SBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto isend(
    SendMode&&, Request&& request, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD
) -> SBuf {
    auto isend_impl = [](auto&&... args) {
        if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::standard_t>) {
            kamping::core::isend(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::buffered_t>) {
            kamping::core::ibsend(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::sync_t>) {
            kamping::core::issend(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::ready_t>) {
            kamping::core::irsend(std::forward<decltype(args)>(args)...);
        }
    };
    isend_impl(sbuf, std::move(dest), std::move(tag), comm, std::forward<Request>(request));
    return std::forward<SBuf>(sbuf);
}
/// High-level overload: creates and owns the MPI_Request internally.
/// Returns iresult<SBuf> which co-locates the request and buffer;
/// call wait() to block and retrieve the buffer.
template <
    is_send_mode                                SendMode,
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto isend(SendMode&&, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> iresult<SBuf> {
    iresult<SBuf> res{std::forward<SBuf>(sbuf)};
    if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::standard_t>) {
        kamping::core::isend(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::buffered_t>) {
        kamping::core::ibsend(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::sync_t>) {
        kamping::core::issend(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
    } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::ready_t>) {
        kamping::core::irsend(res.view(), dest, tag, comm, res.mpi_native_handle_ptr());
    }
    return res;
}

/// Convenience overload: standard mode, external request.
template <
    ranges::send_buffer                                SBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto isend(Request&& request, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> SBuf {
    return isend(
        send_mode::standard,
        std::forward<Request>(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        std::move(tag),
        comm
    );
}

/// Convenience overload: standard mode, internally managed request (returns iresult).
template <
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto isend(SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD) -> iresult<SBuf> {
    return isend(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), std::move(tag), comm);
}

} // namespace kamping::v2
