#pragma once

#include <cstddef>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/status.hpp"
#include "kamping/v2/views/concepts.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "mpi/collectives/allgather.hpp"
#include "mpi/collectives/alltoall.hpp"
#include "mpi/collectives/bcast.hpp"
#include "mpi/collectives/gather.hpp"
#include "mpi/collectives/scatter.hpp"
#include "mpi/comm.hpp"
#include "mpi/p2p/mprobe.hpp"
#include "mpi/p2p/sendrecv.hpp"

/// @file
/// infer() is a customization point that transfers metadata from the sending to the receiving
/// side before an MPI operation is issued. The default behavior sets the recv count on resizable
/// recv buffers. Users can provide their own overloads via ADL for custom buffer types or to
/// transfer additional metadata alongside the count.
///
/// Dispatch is on operation tag types (comm_op::recv, comm_op::allgather, ...) rather than an enum, so
/// users can add new tags without modifying this header.

namespace kamping {

// ---- Default infer() overloads ----------------------------------------------

template <mpi::experimental::recv_buffer RBuf>
auto infer(comm_op::recv, RBuf& rbuf, int source, int tag, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        v2::status  status;
        MPI_Message message = MPI_MESSAGE_NULL;
        mpi::experimental::mprobe(source, tag, comm, message, status);
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(status.count(mpi::experimental::type(rbuf))));
        return message;
    }
}

template <mpi::experimental::send_recv_buffer SRBuf>
auto infer(comm_op::bcast, SRBuf& srbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<SRBuf>) {
        mpi::experimental::comm_view cv{comm};
        int send_count = cv.rank() == root ? static_cast<int>(mpi::experimental::count(srbuf)) : 0;
        mpi::experimental::bcast(v2::views::ref_single(send_count), root, comm);
        if (cv.rank() != root) {
            srbuf.set_recv_count(static_cast<std::ptrdiff_t>(send_count));
        }
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::allgather, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        int comm_size = mpi::experimental::comm_view{comm}.size();
        rbuf.set_recv_count(comm_size * static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer_v RBuf>
void infer(comm_op::allgatherv, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf_v<RBuf>) {
        int comm_size  = mpi::experimental::comm_view{comm}.size();
        int send_count = static_cast<int>(mpi::experimental::count(sbuf));
        rbuf.set_comm_size(comm_size);
        mpi::experimental::allgather(v2::views::ref_single(send_count), mpi::experimental::counts(rbuf), comm);
        rbuf.commit_counts();
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::alltoall, SBuf const& sbuf, RBuf& rbuf, MPI_Comm /* comm */) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
    }
}

template <mpi::experimental::send_buffer_v SBuf, mpi::experimental::recv_buffer_v RBuf>
void infer(comm_op::alltoallv, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf_v<RBuf>) {
        int comm_size = mpi::experimental::comm_view{comm}.size();
        rbuf.set_comm_size(comm_size);
        mpi::experimental::alltoall(mpi::experimental::counts(sbuf), mpi::experimental::counts(rbuf), comm);
        rbuf.commit_counts();
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(
    comm_op::sendrecv, SBuf const& sbuf, RBuf& rbuf, int dest, int send_tag, int source, int recv_tag, MPI_Comm comm
) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        int send_count = static_cast<int>(mpi::experimental::count(sbuf));
        int recv_count = 0;
        mpi::experimental::sendrecv(
            v2::views::ref_single(send_count),
            dest,
            send_tag,
            v2::views::ref_single(recv_count),
            source,
            recv_tag,
            comm,
            MPI_STATUS_IGNORE
        );
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(recv_count));
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::reduce, SBuf const& sbuf, RBuf& rbuf, MPI_Op, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if (mpi::experimental::comm_view{comm}.rank() == root && mpi::experimental::ptr(sbuf) != MPI_IN_PLACE) {
            rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::allreduce, SBuf const& sbuf, RBuf& rbuf, MPI_Op, MPI_Comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if (mpi::experimental::ptr(sbuf) != MPI_IN_PLACE) {
            rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::gather, SBuf const& sbuf, RBuf& rbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        mpi::experimental::comm_view cv{comm};
        if (cv.rank() == root) {
            rbuf.set_recv_count(cv.size() * static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer_v RBuf>
void infer(comm_op::gatherv, SBuf const& sbuf, RBuf& rbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf_v<RBuf>) {
        mpi::experimental::comm_view cv{comm};
        int my_count = static_cast<int>(mpi::experimental::count(sbuf));
        if (cv.rank() == root) {
            rbuf.set_comm_size(cv.size());
        }
        mpi::experimental::gather(v2::views::ref_single(my_count), mpi::experimental::counts(rbuf), root, comm);
        if (cv.rank() == root) {
            rbuf.commit_counts();
        }
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::scatter, SBuf const& sbuf, RBuf& rbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        mpi::experimental::comm_view cv{comm};
        int per_rank_count = static_cast<int>(mpi::experimental::count(sbuf)) / cv.size();
        mpi::experimental::bcast(v2::views::ref_single(per_rank_count), root, comm);
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(per_rank_count));
    }
}

template <mpi::experimental::send_buffer_v SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::scatterv, SBuf const& sbuf, RBuf& rbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        int recv_count = 0;
        mpi::experimental::scatter(mpi::experimental::counts(sbuf), v2::views::ref_single(recv_count), root, comm);
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(recv_count));
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::scan, SBuf const& sbuf, RBuf& rbuf, MPI_Op, MPI_Comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if (mpi::experimental::ptr(sbuf) != MPI_IN_PLACE) {
            rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::exscan, SBuf const& sbuf, RBuf& rbuf, MPI_Op, MPI_Comm) {
    if constexpr (kamping::v2::deferred_recv_buf<RBuf>) {
        if (mpi::experimental::ptr(sbuf) != MPI_IN_PLACE) {
            rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
        }
    }
}

} // namespace kamping
