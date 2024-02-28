#pragma once

#include "kamping/collectives/allreduce.hpp"
#include "kamping/communicator.hpp"

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename, template <typename...> typename>
    typename... Plugins>
size_t kamping::Communicator<DefaultContainerType, Plugins...>::num_numa_nodes() const {
    // Split this communicator into NUMA nodes.
    Communicator numa_comm = split_to_shared_memory();

    // Determine the lowest rank on each NUMA node.
    size_t const numa_representative = numa_comm.allreduce_single(send_buf(rank()), op(ops::min<>{}));

    // Determine the number of NUMA nodes by counting the number of distinct lowest ranks.
    size_t const num_numa_nodes =
        allreduce_single(send_buf(numa_representative == rank() ? 1ul : 0ul), op(ops::plus<>{}));

    return num_numa_nodes;
}
