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

#include "../test_assertions.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/sparse_alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/span.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(AlltoallvSparseTest, alltoall_single_element) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    Communicator comm;

    using msg_type = std::vector<size_t>;

    // Prepare send buffer
    std::vector<std::pair<int, msg_type>> input(comm.size());
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace_back(static_cast<int>(i), msg_type{i});
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](ProbedMessage<size_t, Communicator<>> const& probed_msg) {
        const int source   = probed_msg.source_signed();
        msg_type  recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), 1);
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count_signed());
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count());
        EXPECT_EQ(probed_msg.source(), probed_msg.source_signed());
        sources.emplace_back(source);
        recv_buf.emplace_back(recv_msg.front());
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_EQ(sources, iota_container_n(comm.size(), 0)); // recv message from all ranks
    EXPECT_THAT(recv_buf, Each(comm.rank()));
}

TEST(AlltoallvSparseTest, alltoall_single_element_map_as_send_buf) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    Communicator comm;

    using msg_type = std::vector<size_t>;

    // Prepare send buffer
    std::map<int, msg_type> input;
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace(static_cast<int>(i), msg_type{i});
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](ProbedMessage<size_t, Communicator<>> const& probed_msg) {
        const int source   = probed_msg.source_signed();
        msg_type  recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), 1);
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count_signed());
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count());
        EXPECT_EQ(probed_msg.source(), probed_msg.source_signed());
        sources.emplace_back(source);
        recv_buf.emplace_back(recv_msg.front());
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_EQ(sources, iota_container_n(comm.size(), 0)); // recv message from all ranks
    EXPECT_THAT(recv_buf, Each(comm.rank()));
}

TEST(AlltoallvSparseTest, alltoall_single_element_unordered_map_as_send_buf) {
    // Sends a single element from each rank to each other rank with only the mandatory parameters
    Communicator comm;

    using msg_type = std::vector<size_t>;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace(static_cast<int>(i), msg_type{i});
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](ProbedMessage<size_t, Communicator<>> const& probed_msg) {
        const int source   = probed_msg.source_signed();
        msg_type  recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), 1);
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count_signed());
        EXPECT_EQ(recv_msg.size(), probed_msg.recv_count());
        EXPECT_EQ(probed_msg.source(), probed_msg.source_signed());
        sources.emplace_back(source);
        recv_buf.emplace_back(recv_msg.front());
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_EQ(sources, iota_container_n(comm.size(), 0)); // recv message from all ranks
    EXPECT_THAT(recv_buf, Each(comm.rank()));
}

TEST(AlltoallvSparseTest, one_to_all) {
    // Sends a message from rank 0 to all other ranks
    Communicator comm;

    using msg_type        = std::vector<size_t>;
    const size_t msg_size = 5;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    if (comm.rank() == 0) {
        for (size_t i = 0; i < comm.size(); ++i) {
            input.emplace(static_cast<int>(i), msg_type(msg_size, i));
        }
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](ProbedMessage<size_t, Communicator<>> const& probed_msg) {
        const int source   = probed_msg.source_signed();
        msg_type  recv_msg = probed_msg.recv();
        EXPECT_EQ(recv_msg.size(), msg_size);
        sources.emplace_back(source);
        recv_buf = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_THAT(sources, ElementsAre(0)); // only recv message from rank 0
    EXPECT_EQ(recv_buf, msg_type(msg_size, comm.rank()));
}

TEST(AlltoallvSparseTest, one_to_all_recv_type_out) {
    // Sends a message from rank 0 to all other ranks
    Communicator comm;

    using msg_type        = std::vector<size_t>;
    const size_t msg_size = 5;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    if (comm.rank() == 0) {
        for (size_t i = 0; i < comm.size(); ++i) {
            input.emplace(static_cast<int>(i), msg_type(msg_size, i));
        }
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](ProbedMessage<size_t, Communicator<>> const& probed_msg) {
        const int source                 = probed_msg.source_signed();
        const auto [recv_msg, recv_type] = probed_msg.recv(recv_type_out());

        EXPECT_EQ(recv_msg.size(), msg_size);
        EXPECT_THAT(possible_mpi_datatypes<size_t>(), Contains(recv_type));
        sources.emplace_back(source);
        recv_buf = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_THAT(sources, ElementsAre(0)); // only recv message from rank 0
    EXPECT_EQ(recv_buf, msg_type(msg_size, comm.rank()));
}

TEST(AlltoallvSparseTest, one_to_all_recv_type_out_other_order) {
    // Sends a message from rank 0 to all other ranks
    Communicator comm;

    using msg_type        = std::vector<size_t>;
    const size_t msg_size = 5;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    if (comm.rank() == 0) {
        for (size_t i = 0; i < comm.size(); ++i) {
            input.emplace(static_cast<int>(i), msg_type(msg_size, i));
        }
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](ProbedMessage<size_t, Communicator<>> const& probed_msg) {
        const int source = probed_msg.source_signed();
        const auto [recv_type, recv_msg] = probed_msg.recv(recv_type_out(), kamping::recv_buf(alloc_new<msg_type>));

        EXPECT_EQ(recv_msg.size(), msg_size);
        EXPECT_THAT(possible_mpi_datatypes<size_t>(), Contains(recv_type));
        sources.emplace_back(source);
        recv_buf = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    std::sort(sources.begin(), sources.end());
    EXPECT_THAT(sources, ElementsAre(0)); // only recv message from rank 0
    EXPECT_EQ(recv_buf, msg_type(msg_size, comm.rank()));
}

TEST(AlltoallvSparseTest, one_to_all_owning_send_buf_and_non_owning_recv_buf) {
    // Sends a message from rank 0 to all other ranks
    Communicator comm;

    using msg_type        = std::vector<size_t>;
    const size_t msg_size = 5;

    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    for (size_t i = 0; i < comm.size(); ++i) {
        input.emplace(static_cast<int>(i), msg_type(msg_size, i));
    }

    // Prepare cb
    std::vector<size_t> recv_buf;
    std::vector<int>    sources;
    auto                on_msg = [&](ProbedMessage<size_t, Communicator<>> const& probed_msg) {
        msg_type recv_msg;
        probed_msg.recv(kamping::recv_buf<resize_to_fit>(recv_msg));
        EXPECT_EQ(recv_msg.size(), msg_size);
        recv_buf = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(std::move(input)), on_message(on_msg));

    EXPECT_EQ(recv_buf, msg_type(msg_size, comm.rank()));
}

TEST(AlltoallvSparseTest, sparse_exchange) {
    // Send a message to left and right partner
    Communicator comm;

    if (comm.size() < 2) {
        return;
    }

    using msg_type = std::vector<size_t>;

    int const left_partner  = (comm.size_signed() + comm.rank_signed() - 1) % comm.size_signed();
    int const right_partner = (comm.rank_signed() + 1) % comm.size_signed();
    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    input.emplace(left_partner, msg_type(42, comm.rank()));
    input.emplace(right_partner, msg_type(42, comm.rank()));

    // Prepare cb
    std::unordered_map<int, msg_type> recv_buf;
    std::vector<int>                  sources;
    auto                              on_msg = [&](ProbedMessage<size_t, Communicator<>> const& probed_msg) {
        auto recv_msg                        = probed_msg.recv();
        recv_buf[probed_msg.source_signed()] = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    EXPECT_EQ(recv_buf.size(), 2);
    EXPECT_EQ(recv_buf[left_partner], msg_type(42, asserting_cast<size_t>(left_partner)));
    EXPECT_EQ(recv_buf[right_partner], msg_type(42, asserting_cast<size_t>(right_partner)));
}

TEST(AlltoallvSparseTest, sparse_exchange_custom_dynamic_datatype) {
    // Send a message to left and right partner
    Communicator comm;

    if (comm.size() < 2) {
        return;
    }

    int const msg_count = 42;
    using msg_type      = std::vector<int>;

    int const left_partner  = (comm.size_signed() + comm.rank_signed() - 1) % comm.size_signed();
    int const right_partner = (comm.rank_signed() + 1) % comm.size_signed();
    // Prepare send buffer
    std::unordered_map<int, msg_type> input;
    input.emplace(left_partner, msg_type(msg_count, comm.rank_signed()));
    input.emplace(right_partner, msg_type(msg_count, comm.rank_signed()));

    MPI_Datatype two_ints;
    MPI_Type_contiguous(2, MPI_INT, &two_ints);
    MPI_Type_commit(&two_ints);
    // Prepare cb
    std::unordered_map<int, msg_type> recv_buf;
    std::vector<int>                  sources;
    auto                              on_msg = [&](ProbedMessage<int, Communicator<>> const& probed_msg) {
        msg_type recv_msg(msg_count);
        probed_msg.recv(kamping::recv_buf(recv_msg), recv_type(two_ints));
        EXPECT_EQ(probed_msg.recv_count(two_ints), msg_count / 2);
        EXPECT_EQ(probed_msg.recv_count(), msg_count);
        recv_buf[probed_msg.source_signed()] = recv_msg;
    };

    comm.alltoallv_sparse(sparse_send_buf(input), on_message(on_msg));

    EXPECT_EQ(recv_buf.size(), 2);
    EXPECT_EQ(recv_buf[left_partner], msg_type(msg_count, left_partner));
    EXPECT_EQ(recv_buf[right_partner], msg_type(msg_count, right_partner));

    MPI_Type_free(&two_ints);
}
