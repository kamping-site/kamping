#pragma once
#include "kamping/comm_helper/generic_helper.hpp"

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
        rbuf.set_size(comm);
    }
}

} // namespace kamping