#include <numeric>
#include <thread>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugin/alltoall_grid.hpp"

using namespace ::kamping;

int main(int argc, char** argv) {
    // Call MPI_Init() and MPI_Finalize() automatically.
    Environment<> env(argc, argv);

    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();
    std::vector<double> input(comm.size(), static_cast<double>(comm.rank_signed()) + 0.5);
    std::vector<int>    counts(comm.size(), 1);
    {
        // use grid alltoall with an envelope for each message
        constexpr auto envelope_level = plugin::MessageEnvelopeLevel::source_and_destination;
        auto recv_buf = grid_comm.alltoallv_with_envelope<envelope_level>(send_buf(input), send_counts(counts));
        for (auto const& elem: recv_buf) {
            std::cout << "Received " << elem.get_payload() << " from rank " << elem.get_source() << std::endl;
        }
    }
    {
        // use grid alltoall with conventional api
        [[maybe_unused]] auto [recv_buf, recv_counts] =
            grid_comm.alltoallv(send_buf(input), send_counts(counts), recv_counts_out());
    }

    return EXIT_SUCCESS;
}
