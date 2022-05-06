// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <iostream>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/parameter_factories.hpp"
#include "kamping/parameter_objects.hpp"

struct my_plus {
    template <typename T>
    auto operator()(T a, T b) {
        return a + b;
    }
};

int main() {
    using namespace kamping;

    Environment         e;
    Communicator        comm;
    std::vector<double> input = {1, 2, 3};
    std::vector<double> output;

    auto result0 = comm.reduce(send_buf(input), op(ops::plus<>()), root(0)).extract_recv_buffer();
    print_result_on_root(result0, comm);
    auto result1 = comm.reduce(send_buf(input), op(ops::plus<double>())).extract_recv_buffer();
    print_result_on_root(result1, comm);
    auto result2 = comm.reduce(send_buf(input), op(my_plus{}, commutative)).extract_recv_buffer();
    print_result_on_root(result2, comm);

    auto result3 [[maybe_unused]] =
        comm.reduce(send_buf(input), recv_buf(output), op([](auto a, auto b) { return a + b; }, non_commutative));
    print_result_on_root(output, comm);

    std::vector<std::pair<int, double>> input2 = {{3, 0.25}};

    auto result4 = comm.reduce(
                           send_buf(input2), op(
                                                 [](auto a, auto b) {
                                                     // dummy
                                                     return std::pair(a.first + b.first, a.second + b.second);
                                                 },
                                                 commutative))
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

    auto result5 = comm.reduce(send_buf(input3), op(ops::max<>(), commutative)).extract_recv_buffer();
    if (comm.rank() == 0) {
        for (auto& elem: result5) {
            std::cout << elem.x << " " << elem.y << " " << elem.z << std::endl;
        }
    }

    return 0;
}
