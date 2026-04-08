#pragma once

#include <mpi.h>

#include "kamping/request.hpp"
#include "kamping/v2/error_handling.hpp"
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
template <
    is_send_mode                                       SendMode,
    ranges::send_buffer                                SBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto isend(
    SendMode&&, Request&& request, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD
) -> ranges::buf_result_t<SBuf> {
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
    if constexpr (!std::is_reference_v<SBuf> && !ranges::borrowed_buffer<SBuf>) {
        auto buf = std::move(sbuf);
        isend_impl(buf, std::move(dest), std::move(tag), comm, std::move(request));
        return buf; // NRVO
    } else {
        isend_impl(sbuf, std::move(dest), std::move(tag), comm, std::move(request));
        return std::forward<SBuf>(sbuf);
    }
}

template <
    is_send_mode                                SendMode,
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto isend(SendMode&&, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> ranges::buf_result_t<SBuf> {
      // isend(SendMode{}, std::forward)
}

template <
    ranges::send_buffer                                SBuf,
    bridge::mpi_rank                                   Dest    = int,
    bridge::mpi_tag                                    Tag     = int,
    bridge::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    bridge::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
auto isend(Request&& request, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> ranges::buf_result_t<SBuf> {
    return send(
        send_mode::standard,
        std::move(request),
        std::forward<SBuf>(sbuf),
        std::move(dest),
        std::move(tag),
        comm
    );
}
} // namespace kamping::v2
