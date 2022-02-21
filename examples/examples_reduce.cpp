#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_factories.hpp"
#include <iostream>
#include <mpi.h>
#include <vector>

template <typename T>
void print_result(std::vector<T>& result, kamping::Communicator comm) {
    if (comm.rank() == 0) {
        for (auto elem: result) {
            std::cout << elem << std::endl;
        }
    }
}
struct my_plus {
    template <typename T>
    auto operator()(T a, T b) {
        return a + b;
    }
};

int main() {
    MPI_Init(NULL, NULL);
    kamping::Reduce       reducer;
    kamping::Communicator comm;
    std::vector<double>   input = {1, 2, 3};
    std::vector<double>   output;
    using namespace kamping;

    auto result0 =
        reducer.reduce(comm, kamping::send_buf(input), kamping::op(kamping::ops::plus<>())).extract_recv_buffer();
    print_result(result0, comm);
    auto result1 =
        reducer.reduce(comm, kamping::send_buf(input), kamping::op(kamping::ops::plus<int>())).extract_recv_buffer();
    print_result(result1, comm);
    auto result2 = reducer.reduce(comm, kamping::send_buf(input), kamping::op(my_plus{})).extract_recv_buffer();
    print_result(result2, comm);
    reducer.reduce(
        comm, kamping::send_buf(input), kamping::recv_buf(output), kamping::op([](auto a, auto b) { return a + b; }));
    print_result(output, comm);

    MPI_Finalize();
    return 0;
}
