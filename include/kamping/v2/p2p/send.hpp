#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/p2p/constants.hpp"
#include "kamping/v2/p2p/send_mode.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"

namespace kamping::core {
template <
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void send(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Send(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bsend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Bsend(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void ssend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Ssend(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void rsend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Rsend(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::bridge::to_rank(dest),
        kamping::bridge::to_tag(tag),
        kamping::bridge::native_handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core

namespace kamping::v2 {

template <
    is_send_mode                                SendMode,
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto send(SendMode&&, SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> ranges::buf_result_t<SBuf> {
    auto send_impl = [](auto&&... args) {
        if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::standard_t>) {
            kamping::core::send(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::buffered_t>) {
            kamping::core::bsend(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::sync_t>) {
            kamping::core::ssend(std::forward<decltype(args)>(args)...);
        } else if constexpr (std::same_as<std::decay_t<SendMode>, send_mode::ready_t>) {
            kamping::core::rsend(std::forward<decltype(args)>(args)...);
        }
    };
    if constexpr (!std::is_reference_v<SBuf> && !ranges::borrowed_buffer<SBuf>) {
        auto buf = std::move(sbuf);
        send_impl(buf, std::move(dest), std::move(tag), comm);
        return buf; // NRVO
    } else {
        send_impl(sbuf, std::move(dest), std::move(tag), comm);
        return std::forward<SBuf>(sbuf);
    }
}

template <
    ranges::send_buffer                         SBuf,
    bridge::mpi_rank                            Dest = int,
    bridge::mpi_tag                             Tag  = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto send(SBuf&& sbuf, Dest dest, Tag tag = DEFAULT_SEND_TAG, Comm const& comm = MPI_COMM_WORLD)
    -> ranges::buf_result_t<SBuf> {
    return send(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), std::move(tag), comm);
}

template <ranges::send_buffer SBuf, bridge::mpi_rank Dest = int, bridge::convertible_to_mpi_handle<MPI_Comm> Comm>
auto send(SBuf&& sbuf, Dest dest, Comm const& comm) -> ranges::buf_result_t<SBuf> {
    return send(send_mode::standard, std::forward<SBuf>(sbuf), std::move(dest), DEFAULT_SEND_TAG, comm);
}
} // namespace kamping::v2
