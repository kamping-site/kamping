#pragma once

#include <cstddef>

#include <mpi.h>

#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"

/// @file
/// infer() is a customization point that transfers metadata from the sending to the receiving
/// side before an MPI operation is issued. The default behavior sets the recv count on resizable
/// recv buffers. Users can provide their own overloads via ADL for custom buffer types or to
/// transfer additional metadata alongside the count.
///
/// Dispatch is on operation tag types (comm_op::recv, comm_op::allgather, ...) rather than an enum, so
/// users can add new tags without modifying this header.

namespace kamping {

// ---- Operation tags ---------------------------------------------------------
// Each tag is a distinct empty type used for tag-dispatch in infer() overloads.
// They live in kamping::comm_op:: to avoid clashing with the kamping::recv_tag()
// named-parameter factory in named_parameters.hpp.

namespace comm_op {
struct recv {};
struct allgather {};
struct alltoall {};
struct sendrecv {};
} // namespace comm_op

// ---- Default infer() overloads ----------------------------------------------

template <kamping::ranges::recv_buffer RBuf, typename Comm>
void infer(comm_op::recv, RBuf& rbuf, int source, int tag, Comm const& comm) {
    if constexpr (kamping::ranges::resizable_recv_buf<RBuf>) {
        MPI_Status status;
        MPI_Probe(source, tag, comm.mpi_communicator(), &status);
        int count;
        MPI_Get_count(&status, kamping::ranges::type(rbuf), &count);
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(count));
    }
}

template <kamping::ranges::send_buffer SBuf, kamping::ranges::recv_buffer RBuf, typename Comm>
void infer(comm_op::allgather, SBuf const& sbuf, RBuf& rbuf, Comm const& comm) {
    if constexpr (kamping::ranges::resizable_recv_buf<RBuf>) {
        rbuf.set_recv_count(comm.size() * static_cast<std::ptrdiff_t>(kamping::ranges::size(sbuf)));
    }
}

template <kamping::ranges::send_buffer SBuf, kamping::ranges::recv_buffer RBuf, typename Comm>
void infer(comm_op::alltoall, SBuf const& sbuf, RBuf& rbuf, Comm const& /* comm */) {
    if constexpr (kamping::ranges::resizable_recv_buf<RBuf>) {
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(kamping::ranges::size(sbuf)));
    }
}

template <kamping::ranges::send_buffer SBuf, kamping::ranges::recv_buffer RBuf, typename Comm>
void infer(comm_op::sendrecv, SBuf const& sbuf, RBuf& rbuf, int dest, int source, Comm const& comm) {
    if constexpr (kamping::ranges::resizable_recv_buf<RBuf>) {
        int const send_count = static_cast<int>(kamping::ranges::size(sbuf));
        int       recv_count = 0;
        MPI_Sendrecv(
            &send_count,
            1,
            MPI_INT,
            dest,
            MPI_ANY_TAG,
            &recv_count,
            1,
            MPI_INT,
            source,
            MPI_ANY_TAG,
            comm.mpi_communicator(),
            MPI_STATUS_IGNORE
        );
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(recv_count));
    }
}

} // namespace kamping
