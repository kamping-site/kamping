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
    kamping::Communicator comm;
    std::vector<double>   input = {1, 2, 3};
    std::vector<double>   output;
    using namespace kamping;

    auto my_send_buf = send_buf(input);
    auto result0     = comm.reduce(my_send_buf, op(ops::plus<>()), root(0)).extract_recv_buffer();
    // print_result(result0, comm);
    auto result1 = comm.reduce(my_send_buf, op(ops::plus<double>())).extract_recv_buffer();
    // print_result(result1, comm);
    auto result2 = comm.reduce(my_send_buf, kamping::op(my_plus{}, commutative())).extract_recv_buffer();
    // print_result(result2, comm);

    auto result3 [[maybe_unused]] = comm.reduce(
        my_send_buf, kamping::recv_buf(output), kamping::op([](auto a, auto b) { return a + b; }, non_commutative()));
    // print_result(output, comm);

    std::vector<std::pair<int, double>> input2 = {{3, 0.25}};

    auto result4 = comm.reduce(
                           send_buf(input2), kamping::op(
                                                 [](auto a, auto b) {
                                                     // dummy
                                                     return std::pair(a.first + b.first, a.second + b.second);
                                                 },
                                                 commutative()))
                       .extract_recv_buffer();
    if (comm.rank() == 0) {
        for (auto& elem: result4) {
            std::cout << elem.first << " " << elem.second << std::endl;
        }
    }
    struct Point {
        int           x;
        double        y;
        unsigned long z;
        Point         operator+(Point& rhs) const {
            return {x + rhs.x, y + rhs.y, z + rhs.z};
        }
        bool operator<(Point const& rhs) const {
            return x < rhs.x || (x == rhs.x && y < rhs.y) || (x == rhs.x && y == rhs.y && z < rhs.z);
        }
    };
    std::vector<Point> input3 = {{3, 0.25, 300}, {4, 0.1, 100}};
    if (comm.rank() == 2) {
        input3[1].y = 0.75;
    }

    auto result5 = comm.reduce(send_buf(input3), kamping::op(ops::max<>(), commutative())).extract_recv_buffer();
    if (comm.rank() == 0) {
        for (auto& elem: result5) {
            std::cout << elem.x << " " << elem.y << " " << elem.z << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
