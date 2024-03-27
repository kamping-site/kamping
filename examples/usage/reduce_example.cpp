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
#include "kamping/data_buffer.hpp"
#include "kamping/environment.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"

int main() {
    using namespace kamping;

    Environment  e;
    Communicator comm;

    std::vector<double> input = {1, 2, 3};

    // Compute the sum of all elements scattered across all ranks and store it on the root rank.
    auto const result0 = comm.reduce(send_buf(input), op(ops::plus<>()));

    if (comm.rank() == 0) {
        std::cout << " --- basic --- " << std::endl;
    }
    print_result_on_root(result0, comm);
    // MPI_SUM (with performance penalty) and std::plus<> are also valid predefined operators.

    // Provide a custom commutative reduction function; store the result in a preallocated container.
    std::vector<double> result2(3, 0);
    comm.reduce(send_buf(input), recv_buf(result2), op([](auto a, auto b) { return a + b; }, ops::non_commutative));

    if (comm.rank() == 0) {
        std::cout << " --- custom reduction function --- " << std::endl;
    }
    print_result_on_root(result2, comm);

    // Compute the reduction over a custom datatype using a custom operation.
    struct Bar {
        int    first;
        double second;
    };
    std::vector<Bar> input2 = {{3, 0.25}};

    [[maybe_unused]] auto const result4 = comm.reduce(
        send_buf(input2),
        op(
            [](auto a, auto b) {
                return Bar{a.first + b.first, a.second + b.second};
            },
            ops::commutative
        )
    );

    if (comm.rank() == 0) {
        std::cout << " --- custom datatype, custom function --- " << std::endl;
        for (auto& elem: result4) {
            std::cout << elem.first << " " << elem.second << std::endl;
        }
    }

    // Custom types can also be used with predefined operations given they overload the required members.
    struct Point {
        int           x;
        double        y;
        unsigned long z;

        bool operator<(Point const& rhs) const {
            return x < rhs.x || (x == rhs.x && y < rhs.y) || (x == rhs.x && y == rhs.y && z < rhs.z);
        }
    };

    std::vector<Point> input5 = {{3, 0.25, 300}, {4, 0.1, 100}};
    if (comm.rank() == 2) {
        input5[1].y = 0.75;
    }

    [[maybe_unused]] auto result5 = comm.reduce(send_buf(input5), op(ops::max<>(), ops::commutative));
    if (comm.rank() == 0) {
        std::cout << " --- custom datatype, predefined function --- " << std::endl;
        for (auto& elem: result5) {
            std::cout << elem.x << " " << elem.y << " " << elem.z << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
