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

#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/send.hpp"

/// @brief A plugin providing a function to send the integer 42 to a target rank.
template <typename Comm>
class Send42Plugin {
public:
    /// @brief Sends the single integer `42` to targetRank.
    /// @param targetRank The rank to send 42 to.
    void send42(size_t targetRank) {
        int const send_buf = 42;
        // Use the built-in send function.
        static_cast<Comm&>(*this).send(kamping::send_buf(send_buf), kamping::destination(targetRank));
    }
};

TEST(PluginsTest, additional_function) {
    // Create a new communicator. The first template argument is the default container type (has to be provided when
    // using plugins). The following template arguments are plugin classes.
    kamping::Communicator<std::vector, Send42Plugin> comm;

    auto other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        // Use the send42 function from the plugin.
        comm.send42(other_rank);
    } else if (comm.rank() == other_rank) {
        int        msg;
        MPI_Status status;
        MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm.mpi_communicator(), &status);
        ASSERT_EQ(msg, 42);
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    }
}

/// @brief A plugin providing an alternative allreduce function
template <typename Comm>
class AlternativeAllreducePlugin {
public:
    /// @brief Has the same functionality as `kamping::Communicator::allreduce` with the exception that a `recv_buf`
    /// must be passed and there is no return value.
    template <typename... Args>
    void allreduce(Args... args) {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, op, recv_buf),
            KAMPING_OPTIONAL_PARAMETERS()
        );
        // Use the built-in reduce function with every rank as root.
        for (int i = 0; i < static_cast<Comm&>(*this).size_signed(); ++i) {
            static_cast<Comm&>(*this).reduce(kamping::root(i), std::move(args)...);
        }
    }
};

/// @brief Create a new Communicator class that uses the alternative allreduce implementation
class MyComm : public kamping::Communicator<std::vector, AlternativeAllreducePlugin, Send42Plugin> {
public:
    // Use allreduce from AlternativeAllreducePlugin
    using AlternativeAllreducePlugin::allreduce;
};

TEST(PluginsTest, replace_implementation) {
    // This communicator will still use the original allreduce implementation. If we want to use the alternative
    // implementation, we have to make that explicit as in MyComm.
    kamping::Communicator<std::vector, AlternativeAllreducePlugin> faultyComm;

    // This communicator uses the alternative allreduce implementation and also has the send42 function from before.
    MyComm comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.allreduce(kamping::send_buf(input), kamping::op(kamping::ops::plus<>{}), kamping::recv_buf(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}
