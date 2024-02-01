
#include <cstddef>

#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/plugin_helpers.hpp"

using namespace ::kamping;

/// @brief A plugin implementing the \c num_numa_nodes() function.
/// We're using CRTP to inject plugins into the kamping::Communicator class.
template <typename Comm>
class FaultTolerancePlugin : public plugins::PluginBase<Comm, FaultTolerancePlugin> {
public:
    void handle_mpi_error(int error_code, std::string const& function_name) const;
};

template <typename Comm>
void FaultTolerancePlugin<Comm>::handle_mpi_error(int const error_code, std::string const&) const {
    // std::cout << "Calling the fault tolerant error handler" << std::endl;
    throw "I don't like faults, but I am very tolerant to faults";
}

int main(int argc, char** argv) {
    // Call MPI_Init() and MPI_Finalize() automatically.
    Environment<> env(argc, argv);

    // Create a new communicator object with the desired plugins.
    kamping::Communicator<std::vector, FaultTolerancePlugin> comm;

    // Check that our implementation matches the reference implementation and output put the result.
    try {
        comm.send(send_buf(42), destination(0));
    } catch (char const* e) {
        std::cout << "Now handling the fault: ";
        std::cout << e << std::endl;
    }

    return EXIT_SUCCESS;
}
