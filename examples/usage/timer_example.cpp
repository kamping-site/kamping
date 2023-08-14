// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
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
#include <numeric>
#include <random>
#include <vector>

#include <mpi.h>

#include "helpers_for_examples.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/timer/printer.hpp"
#include "kamping/timer/timer.hpp"

int main() {
    using namespace kamping;

    kamping::Environment e;
    Communicator         comm;
    std::vector<int>     input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> output;

    auto spend_some_time = [&]() {
        static std::mt19937                gen((comm.rank() + 17) * 1001);
        std::uniform_int_distribution<int> distrib(1'000, 10'000'000);
        auto                               it = distrib(gen);
        for (volatile int i = 0; i < it; ++i)
            ;
    };
    timer::Timer t;
    t.synchronize_and_start("algorithm");
    for (size_t i = 0; i < 3; ++i) {
        t.synchronize_and_start("round" + std::to_string(i));
        {
            t.synchronize_and_start("preprocessing");
            spend_some_time();
            t.stop();

            t.synchronize_and_start("core_algorithm");
            for (size_t j = 0; j < 5u; ++j) {
                t.start("subroutine");
                spend_some_time();
                t.stop_and_append(
                    {timer::DataAggregationMode::min,
                     timer::DataAggregationMode::max,
                     timer::DataAggregationMode::gather}
                );
            }
            t.stop();
            t.synchronize_and_start("preprocessing");
            spend_some_time();
            t.stop();
        }
        t.stop_and_append();
    }
    t.stop();
    t.evaluate_and_print(kamping::timer::SimpleJsonPrinter{});
    std::cout << std::endl;
    t.evaluate_and_print(kamping::timer::FlatPrinter{});
    std::cout << std::endl;

    return 0;
}
