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
    if constexpr (HasSetSize<RBuff>) {
        size_t recv_size = get_recv_size<type>(sbuf, rbuf, comm);
        rbuf.set_size(recv_size);
    }
}

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
requires(type == CommType::alltoallv) void infer(SBuff& sbuf, RBuff& rbuf, Communicator& comm) {
    KASSERT(
        comm.is_same_on_all_ranks(HasSetSizeV<RBuff>),
        "Receive counts have to be computed on some ranks, but not on all or on none",
        assert::light_communication
    );
    // Calc recv counts
    if constexpr (HasSetSizeV<RBuff>) {
        auto             send_counts = sbuf.size_v();
        std::vector<int> recv_counts(comm.size());
        comm.alltoall(send_counts, recv_counts);
        rbuf.set_size_v(std::move(recv_counts));
    }
}

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
requires(type == CommType::sendrecv) void infer(
    SBuff& sbuf, RBuff& rbuf, int source, int destination, Communicator& comm
) {
    // Use sendrecv to exchange the recv sizes
    KASSERT(
        comm.is_same_on_all_ranks(HasSetSize<RBuff>),
        "Receive count has to be computed on some ranks, but not on all or on none",
        assert::light_communication
    );
    if constexpr (HasSetSize<RBuff>) {
        size_t                           send_size = std::ranges::size(sbuf);
        std::ranges::single_view<size_t> send_buff(send_size);
        std::ranges::single_view<size_t> recv_buff(0);
        comm.sendrecv(send_buff, recv_buff, destination, 0, source);
        rbuf.set_size(*recv_buff.data());
    }
}

template <CommType type, typename RBuff, typename Communicator>
requires(type == CommType::recv) void infer(RBuff& rbuf, Communicator& comm) {
    // RBuff has set_size -> assume it's size is not correct, probe for the actual recv size
    if constexpr (HasSetSize<RBuff>) {
        auto   status = comm.probe(status_out()).extract_status();
        size_t size   = kamping::asserting_cast<size_t>(status.template count_signed<int>());
        rbuf.set_size(size);
    }
}

} // namespace kamping