#include <numeric>
#include <thread>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugins/alltoall_grid_plugin.hpp"

using namespace ::kamping;

int main(int argc, char** argv) {
    // Call MPI_Init() and MPI_Finalize() automatically.
    Environment<> env(argc, argv);

    Communicator<std::vector, plugin::GridCommunicatorPlugin> comm;
    comm.initialize_grid();
    std::vector<double> input(comm.size(), static_cast<double>(comm.rank_signed()) + 0.5);
    std::vector<int>    counts(comm.size(), 1);
    auto recv_buf = comm.alltoallv_grid<plugin::MsgEnvelopeLevel::source_and_destination>(send_buf(input), send_counts(counts));

    using namespace std::literals;
    comm.barrier();
    std::this_thread::sleep_for(10ms);
    comm.barrier();
    if (comm.is_root(0)) {
        for (auto const& elem: recv_buf) {
            std::cout << elem << std::endl;
        }
    }
    comm.barrier();
    std::this_thread::sleep_for(10ms);
    comm.barrier();
    if (comm.is_root(1)) {
        for (auto const& elem: recv_buf) {
            std::cout << elem << std::endl;
        }
    }
    std::this_thread::sleep_for(10ms);
    comm.barrier();
    if (comm.is_root(2)) {
        for (auto const& elem: recv_buf) {
            std::cout << elem << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
