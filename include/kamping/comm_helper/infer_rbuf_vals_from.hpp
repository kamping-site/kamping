#pragma once
#include "kamping/comm_helper/generic_helper.hpp"

namespace kamping {

// Base infer is a no opt
template <CommType type, typename SBuff, typename RBuff, typename Communicator>
void infer(SBuff const&, RBuff const&, Communicator const&) {}

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
    requires HasResize<RBuff>
void infer(SBuff const& sbuf, RBuff& rbuf, Communicator const& comm) {
    size_t recv_size = 0;

    if constexpr (type == CommType::allgather) {
        recv_size = get_recv_size<CommType::allgather>(sbuf, rbuf, comm);
    }

    rbuf.resize(recv_size);

    if constexpr (HasUnderlying<RBuff>) {
        infer(sbuf, rbuf.underlying(), comm);
    }
}

} // namespace kamping