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

#include <gtest/gtest.h>
#include <mpi.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/receive.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;

TEST(RecvTest, return_type) {
    Communicator comm;

    auto int_vector = comm.recv<int>().extract_recv_buffer();
    static_assert(std::is_same_v<decltype(int_vector), std::vector<int>>);

    auto char_owncontainer = comm.recv(recv_buf(NewContainer<testing::OwnContainer<char>>())).extract_recv_buffer();
    static_assert(std::is_same_v<decltype(char_owncontainer), testing::OwnContainer<char>>);
}
