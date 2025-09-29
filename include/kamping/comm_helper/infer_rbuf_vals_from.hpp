#pragma once

#include <ranges>

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/p2p/probe.hpp"

namespace kamping {

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
void infer(SBuff& sbuf, RBuff& rbuf, Communicator& comm) {
    if constexpr (type == CommType::allgather) {
        if constexpr (HasSetSize<RBuff>) {
            size_t recv_size = get_recv_size<CommType::allgather>(sbuf, rbuf, comm);
            rbuf.set_size(recv_size);
        }
    }
}

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
void infer(SBuff& sbuf, RBuff& rbuf, int source, int destination, Communicator& comm) {
    if constexpr (type == CommType::sendrecv) {
        // Use sendrecv to exchange the recv sizes
        if constexpr (HasSetSize<RBuff>) {
            size_t                           send_size = std::ranges::size(sbuf);
            std::ranges::single_view<size_t> send_buff(send_size);
            std::ranges::single_view<size_t> recv_buff(0);
            comm.sendrecv(send_buff, recv_buff, destination, 0, source);
            rbuf.set_size(*recv_buff.data());
        }
    }
}

template <CommType type, typename RBuff, typename Communicator>
void infer(RBuff& rbuf, Communicator& comm) {
    if constexpr (type == CommType::recv) {
        // RBuff has set_size -> assume it's size is not correct, probe for the actual recv size
        if constexpr (HasSetSize<RBuff>) {
            auto   status = comm.probe(status_out()).extract_status();
            size_t size   = kamping::asserting_cast<size_t>(status.template count_signed<int>());
            rbuf.set_size(size);
        }
    }
}

} // namespace kamping