#pragma once

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/p2p/probe.hpp"

namespace kamping {

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
void infer(SBuff& sbuf, RBuff& rbuf, Communicator& comm) {
    if constexpr (type == CommType::allgather) {
        size_t recv_size = get_recv_size<CommType::allgather>(sbuf, rbuf, comm);
        rbuf.set_size(recv_size);
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