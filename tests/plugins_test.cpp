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
#include "kamping/plugin_helpers.hpp"

/// @brief A plugin providing a function to send the integer 42 to a target rank.
template <typename Comm>
class Send42Plugin : public kamping::plugins::PluginBase<Comm, Send42Plugin> {
public:
    /// @brief Sends the single integer `42` to target_rank.
    /// @param target_rank The rank to send 42 to.
    void send42(size_t target_rank) {
        int const send_buf = 42;
        // Use the built-in send function.
        // Uses the `to_communicator` function of `PluginBase` to cast itself to `Comm`.
        this->to_communicator().send(kamping::send_buf(send_buf), kamping::destination(target_rank));
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

/// @brief A plugin providing an alternative allreduce function.
template <typename Comm>
class AlternativeAllreducePlugin : public kamping::plugins::PluginBase<Comm, AlternativeAllreducePlugin> {
public:
    /// @brief Has the same functionality as `kamping::Communicator::allreduce` with the exception that a `recv_buf`
    /// must be passed and there is no return value. Also leaves the recv_buf on rank 0 untouched.
    template <typename... Args>
    void allreduce(Args... args) {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, op, recv_buf),
            KAMPING_OPTIONAL_PARAMETERS()
        );
        // Use the built-in reduce function with every rank as root but skip rank 0.
        // Uses the `to_communicator` function of `PluginBase` to cast itself to `Comm`.
        for (int i = 1; i < this->to_communicator().size_signed(); ++i) {
            this->to_communicator().reduce(kamping::root(i), std::move(args)...);
        }
    }
};

/// @brief Create a new Communicator class that uses the alternative allreduce implementation.
class MyComm : public kamping::Communicator<std::vector, AlternativeAllreducePlugin, Send42Plugin> {
public:
    // Use allreduce from AlternativeAllreducePlugin
    using AlternativeAllreducePlugin::allreduce;
};

TEST(PluginsTest, replace_implementation) {
    // First, a quick example of how NOT to overwrite an existing function:
    {
        // This communicator will still use the original allreduce implementation. If we want to use the alternative
        // implementation, we have to make that explicit as in MyComm.
        kamping::Communicator<std::vector, AlternativeAllreducePlugin> faultyComm;

        std::vector<int> input = {faultyComm.rank_signed(), 42};
        std::vector<int> result;

        // Calling allreduce on this communicator uses the original allreduce implementation.
        faultyComm.allreduce(kamping::send_buf(input), kamping::op(kamping::ops::plus<>{}), kamping::recv_buf(result));

        // On all ranks, the result of the reduce operation is available. Even on rank 0 where the alternative allreduce
        // implementation would leave result unchanged.
        EXPECT_EQ(result.size(), 2);

        std::vector<int> expected_result = {
            (faultyComm.size_signed() * (faultyComm.size_signed() - 1)) / 2,
            faultyComm.size_signed() * 42};
        EXPECT_EQ(result, expected_result);
    }

    // This communicator uses the alternative allreduce implementation and also has the send42 function from before.
    MyComm comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    // Because of the using-declaration in MyComm, this uses the alternative allreduce implementation.
    comm.allreduce(kamping::send_buf(input), kamping::op(kamping::ops::plus<>{}), kamping::recv_buf(result));

    // Check result of the alternative allreduce implementation.
    if (comm.rank() == 0) {
        // On rank 0 result should be unchanged.
        EXPECT_EQ(result.size(), 0);
    } else {
        // On all other ranks, the result of the reduce operation should be available.
        EXPECT_EQ(result.size(), 2);

        std::vector<int> expected_result = {
            (comm.size_signed() * (comm.size_signed() - 1)) / 2,
            comm.size_signed() * 42};
        EXPECT_EQ(result, expected_result);
    }

    // You can also add multiple plugins. MyComm has both AlternativeAllreducePlugin and Send42Plugin so we can use
    // both.
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

/// @brief A plugin providing a function to send a default constructed `T` to a target rank. Use the inner class
/// SendDefaultConstructedPlugin as the actual plugin.
template <typename T>
class SendDefaultConstructedPluginOuterClass {
public:
    /// @brief A plugin providing a function to send a default constructed `T` to a target rank.
    template <typename Comm>
    class SendDefaultConstructedPlugin : public kamping::plugins::PluginBase<Comm, SendDefaultConstructedPlugin> {
    public:
        /// @brief Sends a default constructed `T` to target_rank.
        /// @param target_rank The rank to send to.
        void send_default_constructed(size_t target_rank) {
            T const send_buf{};
            // Use the built-in send function.
            // Uses the `to_communicator` function of `PluginBase` to cast itself to `Comm`.
            this->to_communicator().send(kamping::send_buf(send_buf), kamping::destination(target_rank));
        }
    };
};

TEST(PluginsTest, additional_function_with_double_template) {
    // Create a new communicator. The first template argument is the default container type (has to be provided when
    // using plugins). The following template arguments are plugin classes. Here, we use the inner class
    // `SendDefaultConstructedPlugin` of the outer class `SendDefaultConstructedPluginOuterClass<double>` to send a
    // default constructed `double`.
    kamping::Communicator<std::vector, SendDefaultConstructedPluginOuterClass<double>::SendDefaultConstructedPlugin>
        comm;

    auto other_rank = (comm.root() + 1) % comm.size();
    if (comm.is_root()) {
        // Use the send_default_constructed function from the plugin.
        comm.send_default_constructed(other_rank);
    } else if (comm.rank() == other_rank) {
        double     msg = 3.14;
        MPI_Status status;
        MPI_Recv(&msg, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm.mpi_communicator(), &status);
        double defaultConstructedBool{};
        ASSERT_EQ(msg, defaultConstructedBool);
        ASSERT_EQ(status.MPI_SOURCE, comm.root());
        ASSERT_EQ(status.MPI_TAG, 0);
    }
}
