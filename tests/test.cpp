#include "wrapper.hpp"

#include <iostream>
#include <mpi.h>
#include <sstream>
#include <vector>


void printResult(int rank, std::vector<int> &recvData) {
    std::stringstream ss;
    ss << rank << ": [";
    for(auto elem : recvData) {
        ss << elem << ", ";
    }
    ss << "]" << std::endl;
    std::cout << ss.str() << std::flush;
}

int main() {
    MPI_Init(nullptr, nullptr);
    MPIWrapper::MPIContext ctx{MPI_COMM_WORLD};

    std::vector<int> sendData(ctx.rank() + 1, ctx.rank());
    auto recvDataContainer = ctx.gatherv(in(sendData));
    auto recvData = recvDataContainer.getRecvBuff();

    printResult(ctx.rank(), recvData);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> recvData2;
    ctx.gatherv(in(sendData), out(recvData2), root(1));

    printResult(ctx.rank(), recvData2);

    MPI_Finalize();
}
