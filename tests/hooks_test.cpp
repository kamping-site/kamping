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
#include "kamping/plugin/plugin_helpers.hpp"
#include "kassert/kassert.hpp"

int desired_mpi_ret_code;

// Overwrite MPI_Bcast: Do nothing but return the desired return code.
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

bool first_error_handler_called  = false;
bool second_error_handler_called = false;

/// @brief A plugin overwriting the MPI return code handler.

template <typename Comm, template <typename...> typename DefaultContainerType>
class IgnoreMPIErrors : public kamping::plugin::PluginBase<Comm, DefaultContainerType, IgnoreMPIErrors> {
public:
    void mpi_error_handler([[maybe_unused]] int const ret, [[maybe_unused]] std::string const& callee) const {
        KASSERT(ret != MPI_SUCCESS, "MPI error handler called with MPI_SUCCESS");
        first_error_handler_called = true;
    }
};

template <typename Comm, template <typename...> typename DefaultContainerType>
class IgnoreMPIErrors2 : public kamping::plugin::PluginBase<Comm, DefaultContainerType, IgnoreMPIErrors2> {
public:
    void mpi_error_handler([[maybe_unused]] int const ret, [[maybe_unused]] std::string const& callee) const {
        KASSERT(ret != MPI_SUCCESS, "MPI error handler called with MPI_SUCCESS");
        second_error_handler_called = true;
    }
};

TEST(HooksTest, MPIErrorHook) {
    using namespace kamping;

    Communicator<std::vector, IgnoreMPIErrors> comm;

    size_t value = 0;

    desired_mpi_ret_code       = MPI_SUCCESS;
    first_error_handler_called = false;
    comm.bcast_single(send_recv_buf(value));
    EXPECT_FALSE(first_error_handler_called);

    desired_mpi_ret_code       = MPI_ERR_COMM;
    first_error_handler_called = false;
    comm.bcast_single(send_recv_buf(value));
    EXPECT_TRUE(first_error_handler_called);
}

TEST(HooksTest, TwoPluginsProvidingAnMPIErrorHandler) {
    using namespace kamping;

    Communicator<std::vector, IgnoreMPIErrors, IgnoreMPIErrors2> comm;

    size_t value = 0;

    desired_mpi_ret_code        = MPI_SUCCESS;
    first_error_handler_called  = false;
    second_error_handler_called = false;
    comm.bcast_single(send_recv_buf(value));
    EXPECT_FALSE(first_error_handler_called);
    EXPECT_FALSE(second_error_handler_called);

    desired_mpi_ret_code        = MPI_ERR_COMM;
    first_error_handler_called  = false;
    second_error_handler_called = false;
    comm.bcast_single(send_recv_buf(value));
    EXPECT_TRUE(first_error_handler_called);
    EXPECT_FALSE(second_error_handler_called);
}
