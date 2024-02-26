// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/plugin_helpers.hpp"
#include "kassert/kassert.hpp"

// Overwrite MPI_Bcast: Do nothing but return the desired return code.
int desired_mpi_ret_code  = 0;
int received_mpi_ret_code = 0;

int MPI_Bcast(
    [[maybe_unused]] void*        buffer,
    [[maybe_unused]] int          count,
    [[maybe_unused]] MPI_Datatype datatype,
    [[maybe_unused]] int          root,
    [[maybe_unused]] MPI_Comm     comm
) {
    // Do no call PMPI_Bcast, because we have no need to do something useful.
    return desired_mpi_ret_code;
}

/// @brief A plugin overwriting the MPI return code handler.
template <typename Comm>
class IgnoreMPIErrorsPlugin : public kamping::plugins::PluginBase<Comm, IgnoreMPIErrorsPlugin> {
public:
    void mpi_ret_code_hook(int const ret, [[maybe_unused]] std::string const& function) const {
        received_mpi_ret_code = ret;
    }
};

TEST(HooksTest, MPIRetCode) {
    using namespace kamping;

    Communicator<std::vector, IgnoreMPIErrorsPlugin> comm;

    size_t value = 0;

    desired_mpi_ret_code  = MPI_SUCCESS;
    received_mpi_ret_code = MPI_ERR_COMM;
    comm.bcast_single(send_recv_buf(value));
    EXPECT_EQ(desired_mpi_ret_code, received_mpi_ret_code);

    desired_mpi_ret_code  = MPI_ERR_COMM;
    received_mpi_ret_code = MPI_SUCCESS;
    comm.bcast_single(send_recv_buf(value));
    EXPECT_EQ(desired_mpi_ret_code, received_mpi_ret_code);
}
