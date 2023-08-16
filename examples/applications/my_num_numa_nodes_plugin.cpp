
#include <cstddef>

#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/plugin_helpers.hpp"

using namespace ::kamping;

/// @brief A plugin implementing a simple sparse all-to-all communication.
/// We're using CRTP to inject plugins into the kamping::Communicator class.
template <typename Comm>
class MyNumNumaNodesPlugin : public plugins::PluginBase<Comm, MyNumNumaNodesPlugin> {
public:
    template <typename... Args>
    auto my_num_numa_nodes() const {
        // Split this communicator into NUMA nodes.
        auto numa_comm = this->split_to_numa_nodes();

        // Determine the lowest rank on each NUMA node.
        size_t const numa_representative = numa_comm.allreduce_single(send_buf(this->rank()), op(ops::min<>{}));

        // Determine the number of NUMA nodes by counting the number of distinct lowest ranks.
        size_t const num_numa_nodes =
            this->allreduce_single(send_buf(numa_representative == numa_comm.rank() ? 1 : 0), op(ops::plus<>{}));

        return num_numa_nodes;
    }
};

int main(int argc, char** argv) {
    Environment<>                                            env(argc, argv);
    kamping::Communicator<std::vector, MyNumNumaNodesPlugin> comm;

    KASSERT(comm.my_num_numa_nodes() == comm.num_numa_nodes());
    std::cout << "Number of numa nodes: " << comm.my_num_numa_nodes() << std::endl;

    return EXIT_SUCCESS;
}
