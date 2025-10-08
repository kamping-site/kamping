#pragma once

#include <numeric>
#include <ranges>

#include "kamping/comm_helper/generic_helper.hpp"
#include "kamping/comm_helper/is_same_on_all_ranks.hpp"
#include "kamping/p2p/probe.hpp"

namespace kamping {

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
void infer(SBuff& sbuf, RBuff& rbuf, Communicator& comm) {
    if constexpr (type == CommType::allgather || type == CommType::alltoall) {
        if constexpr (HasSetSize<RBuff>) {
            size_t recv_size = get_recv_size<type>(sbuf, rbuf, comm);
            rbuf.set_size(recv_size);
        }
    }
    if constexpr (type == CommType::alltoallv) {
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
        // Calc send displacement
        if constexpr (HasSetDisplacements<SBuff>) {
            auto             send_counts = sbuf.size_v();
            std::vector<int> send_displacements(comm.size());
            std::exclusive_scan(
                send_counts.begin(),
                send_counts.begin() + asserting_cast<int>(comm.size()),
                send_displacements.begin(),
                0
            );
            sbuf.set_displacements(std::move(send_displacements));
        }
        // Calc recv displacements
        if constexpr (HasSetDisplacements<RBuff>) {
            auto             recv_counts = rbuf.size_v();
            std::vector<int> recv_displacements(comm.size());
            std::exclusive_scan(
                recv_counts.begin(),
                recv_counts.begin() + asserting_cast<int>(comm.size()),
                recv_displacements.begin(),
                0
            );
            rbuf.set_displacements(std::move(recv_displacements));
        }
        // Resize the recv buffer
        if constexpr (HasSetSize<RBuff>) {
            auto recv_displs = rbuf.displacements();
            auto recv_counts = rbuf.size_v();

            int recv_buf_size = 0;
            for (size_t i = 0; i < comm.size(); ++i) {
                recv_buf_size = std::max(recv_buf_size, recv_counts[i] + recv_displs[i]);
            }

            rbuf.set_size(asserting_cast<size_t>(recv_buf_size));
        }
    }
}

template <CommType type, typename SBuff, typename RBuff, typename Communicator>
void infer(SBuff& sbuf, RBuff& rbuf, int source, int destination, Communicator& comm) {
    if constexpr (type == CommType::sendrecv) {
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