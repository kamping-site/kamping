#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

template <typename T>
void print_result(std::vector<T>& result, kamping::Communicator comm) {
    if (comm.rank() == 0) {
        for (auto elem: result) {
            std::cout << elem << std::endl;
        }
    }
}

int main() {
    using namespace kamping;
    MPI_Init(NULL, NULL);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    kamping::Communicator comm;
    std::vector<int>      input(asserting_cast<size_t>(comm.size()));
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> output;

    comm.alltoall(send_buf(input), recv_buf(output));
    print_result(output, comm);

    MPI_Finalize();
    return 0;
}
