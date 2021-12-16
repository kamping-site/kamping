#include "kamping/wrapper.hpp"
#include "kamping/assert.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>


void printResult(int rank, std::vector<int>& recvData, std::string name) {
    std::stringstream ss;
    ss << rank << ": " << name << ": [";
    for (auto elem: recvData) {
        ss << elem << ", ";
    }
    ss << "]" << std::endl;
    std::cout << ss.str() << std::flush;
}

void printResult(int rank, std::unique_ptr<int[]>& recvData, size_t size, std::string name) {
    std::stringstream ss;
    ss << rank << ": " << name << ": [";
    for (size_t i = 0; i < size; ++i) {
        ss << recvData.get()[i] << ", ";
    }
    ss << "]" << std::endl;
    std::cout << ss.str() << std::flush;
}

using namespace kamping;

int main() {
    std::vector<int> a;
    std::vector<int> b;

    ASSERT(false || true && false, "", assert::lightweight);
    std::exit(0);

    MPI_Init(nullptr, nullptr);
    MPIWrapper::MPIContext ctx{MPI_COMM_WORLD};

    std::vector<int> sendData(static_cast<std::size_t>(ctx.rank() + 1), ctx.rank());
    // Gather sendData on PE 0 and allocate all other buffers inside the library
    auto gatherResults = ctx.gatherv(in(sendData));
    auto recvData      = gatherResults.extractRecvBuff();
    auto recvCounts    = gatherResults.extractRecvCounts();
    auto recvDispls    = gatherResults.extractRecvDispls();

    printResult(ctx.rank(), recvData, "data");
    printResult(ctx.rank(), recvCounts, "counts");
    printResult(ctx.rank(), recvDispls, "displs");

    MPI_Barrier(MPI_COMM_WORLD);
    if (ctx.rank() == 0) {
        std::cout << "----------------------------------------------" << std::endl;
    }
    // Sleep so all thread have time to flush their buffers
    std::this_thread::sleep_for(std::chrono::seconds(1));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> recvData2;
    // Gather sendData on PE 1, use existing vector to receive into and return the counts as a new unique_ptr
    auto gatherResults2 = ctx.gatherv(in(sendData), out(recvData2), root(1), recv_counts(new_pointer<int>()));

    // This fails because we have given the out vector as input
    // auto recvData3 = gatherResults2.extractRecvBuff();

    // unique_ptr because we requested it in the gatherv call
    std::unique_ptr<int[]> recvCounts2 = gatherResults2.extractRecvCounts();

    // default is vector
    std::vector<int> recvDispls2 = gatherResults2.extractRecvDispls();

    printResult(ctx.rank(), recvData2, "data");
    printResult(ctx.rank(), recvCounts2, static_cast<std::size_t>(ctx.rank() == 1 ? ctx.size() : 0), "counts");
    printResult(ctx.rank(), recvDispls2, "displs");

    MPI_Finalize();
}
