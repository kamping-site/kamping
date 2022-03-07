#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"
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
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    kamping::Reduce       reducer;
    kamping::Communicator comm;
    std::vector<double>   input = {1, 2, 3};
    std::vector<double>   output;
    using namespace kamping;

    auto my_send_buf = send_buf(input);
    auto result0     = reducer.reduce(comm, my_send_buf, op(ops::plus<>()), root(0)).extract_recv_buffer();
    print_result(result0, comm);
    auto result1 = reducer.reduce(comm, my_send_buf, op(ops::plus<double>())).extract_recv_buffer();
    print_result(result1, comm);
    auto result2 = reducer.reduce(comm, my_send_buf, kamping::op(my_plus{}, commutative())).extract_recv_buffer();
    print_result(result2, comm);

    reducer.reduce(
        comm, my_send_buf, kamping::recv_buf(output),
        kamping::op([](auto a, auto b) { return a + b; }, non_commutative()));
    print_result(output, comm);

    MPI_Finalize();
    return 0;
}
