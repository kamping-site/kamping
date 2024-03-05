
#include <cstddef>

#include <kamping/collectives/allreduce.hpp>
#include <mpi.h>

#include "kamping/comm_helper/num_numa_nodes.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

using namespace ::kamping;

/// @brief A plugin implementing the \c num_numa_nodes() function.
/// We're using CRTP to inject plugins into the kamping::Communicator class.
template <typename Comm, template <typename...> typename DefaultContainerType>
class MyNumNumaNodes : public plugin::PluginBase<Comm, DefaultContainerType, MyNumNumaNodes> {
public:
    /// @brief Number of NUMA nodes (different shared memory regions) in this communicator.
    /// This operation is expensive (communicator splitting and communication). You should cache the result if you need
    /// it multiple times.
    /// @return Number of compute nodes (hostnames) in this communicator.
    size_t my_num_numa_nodes() const;
};

template <typename Comm, template <typename...> typename DefaultContainerType>
size_t MyNumNumaNodes<Comm, DefaultContainerType>::my_num_numa_nodes() const {
    // Uses the \c to_communicator() function of \c PluginBase to cast itself to \c Comm
    auto& self = this->to_communicator();

    // Split this communicator into NUMA nodes.
    auto numa_comm = self.split_to_shared_memory();

    // Determine the lowest rank on each NUMA node.
    size_t const numa_representative = numa_comm.allreduce_single(send_buf(self.rank()), op(ops::min<>{}));

    // Determine the number of NUMA nodes by counting the number of distinct lowest ranks.
    size_t const num_numa_nodes =
        self.allreduce_single(send_buf(numa_representative == numa_comm.rank() ? 1ul : 0), op(ops::plus<>{}));

    return num_numa_nodes;
}

int main(int argc, char** argv) {
    // Call MPI_Init() and MPI_Finalize() automatically.
    Environment<> env(argc, argv);

    // Create a new communicator object with the desired plugins.
    kamping::Communicator<std::vector, MyNumNumaNodes> comm;

    // Check that our implementation matches the reference implementation and output put the result.
    KASSERT(comm.my_num_numa_nodes() == comm.num_numa_nodes());
    std::cout << "Number of numa nodes: " << comm.my_num_numa_nodes() << std::endl;

    return EXIT_SUCCESS;
}
