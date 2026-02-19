#pragma once

#include <numeric>
#include <ranges>

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/p2p/probe.hpp"

namespace kamping {

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
requires(type == CommType::allgather || type == CommType::alltoall) void infer(
    SBuff& sbuf, RBuff& rbuf, Communicator& comm
) {
    if constexpr (ResizableBuffer<RBuff>) {
        size_t recv_size = get_recv_size<type>(sbuf, rbuf, comm);
        rbuf.resize(recv_size);
    }
}

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
requires(type == CommType::alltoallv) void infer(SBuff& sbuf, RBuff& rbuf, Communicator& comm) {
    KASSERT(
        comm.is_same_on_all_ranks(ResizableSizeV<RBuff>),
        "Receive counts have to be computed on some ranks, but not on all or on none",
        assert::light_communication
    );
    // Calc recv counts and resize rbuf.size_v to communicator size if the size_v_resizable_tag tag is set
    if constexpr (ResizableSizeV<RBuff>) {
        auto& recv_counts = rbuf.size_v();
        recv_counts.resize(comm.size());
        comm.alltoall(sbuf.size_v(), recv_counts);
    }
}

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
requires(type == CommType::sendrecv) void infer(
    SBuff& sbuf, RBuff& rbuf, int source, int destination, Communicator& comm
) {
    // Use sendrecv to exchange the recv sizes
    KASSERT(
        comm.is_same_on_all_ranks(ResizableBuffer<RBuff>),
        "Receive count has to be computed on some ranks, but not on all or on none",
        assert::light_communication
    );
    if constexpr (ResizableBuffer<RBuff>) {
        size_t                           send_size = std::ranges::size(sbuf);
        std::ranges::single_view<size_t> send_buff(send_size);
        std::ranges::single_view<size_t> recv_buff(0);
        comm.sendrecv(send_buff, recv_buff, destination, 0, source);
        rbuf.resize(*recv_buff.data());
    }
}

template <CommType type, typename RBuff, typename Communicator>
requires(type == CommType::recv) void infer(RBuff& rbuf, Communicator& comm) {
    if constexpr (ResizableBuffer<RBuff>) {
        auto   status = comm.probe(status_out()).extract_status();
        size_t size   = kamping::asserting_cast<size_t>(status.count_signed(kamping::type(rbuf)));
        rbuf.resize(size);
    }
}

} // namespace kamping